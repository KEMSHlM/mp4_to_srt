import gradio as gr
import os
from pydub import AudioSegment
import openai

# OpenAIのクライアントを初期化
openai.api_key = os.environ["OPENAI_API_KEY"]
client = openai


def clear_tmp_folder():
    """'tmp' フォルダの中身をクリアする"""
    tmp_folder = "tmp"
    os.makedirs(tmp_folder, exist_ok=True)
    for file_name in os.listdir(tmp_folder):
        file_path = os.path.join(tmp_folder, file_name)
        if os.path.isfile(file_path):
            os.unlink(file_path)


def get_duration(inputs, language):
    """音声ファイルを分割し、各チャンクを文字起こししてSRTファイルとトランスクリプトを生成する"""
    audio = AudioSegment.from_file(inputs, format="m4a")
    length = len(audio)
    chunk_length = 1000000  # 1,000,000ミリ秒ごとに分割
    overlap_length = 10000  # 10,000ミリ秒のオーバーラップ
    chunks = []

    # チャンクを生成する際にオーバーラップを考慮
    for i in range(0, length, chunk_length - overlap_length):
        end = min(i + chunk_length, length)
        chunk = audio[i:end]
        chunks.append((chunk, i, end))  # チャンクと実際の開始/終了時間を保存

    all_segments = []
    transcript_text = ""

    for i, (chunk, start, end) in enumerate(chunks, start=1):
        output_file = f"tmp/split{i}.mp3"
        chunk.export(output_file, format="mp3")
        with open(output_file, "rb") as audio_file:
            transcript = client.Audio.transcribe(
                file=audio_file,
                model="whisper-1",
                response_format="verbose_json",
                language=language,
            )
            for segment in transcript.segments:
                segment["start"] += start / 1000  # チャンクの実際の開始時間を加算
                segment["end"] += start / 1000  # チャンクの実際の終了時間を加算
                all_segments.append(segment)
            transcript_text += transcript.text + "\n"

    srt_content = convert_to_srt(all_segments)
    save_srt_file(srt_content, "tmp/output.srt")
    save_srt_file(transcript_text, "tmp/full_transcript.txt")

    return srt_content, transcript_text


def convert_to_srt(segments):
    """セグメントをSRTフォーマットに変換する"""

    def format_time(time_in_seconds):
        """秒をHH:MM:SS,mmmフォーマットに変換する"""
        hours, remainder = divmod(time_in_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02},{milliseconds:03}"

    return "\n".join(
        [
            f"{index + 1}\n{format_time(segment['start'])} --> {format_time(segment['end'])}\n{segment['text']}\n"
            for index, segment in enumerate(segments)
        ]
    )


def save_srt_file(srt_content, file_name):
    """SRTコンテンツをファイルに保存する"""
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(srt_content)


clear_tmp_folder()

# Gradioインターフェイスの設定
iface = gr.Interface(
    fn=get_duration,
    inputs=[
        gr.Audio(type="filepath", label="会議ファイルをアップロード"),
        gr.Dropdown(choices=["en", "ja"], label="言語選択"),
    ],
    outputs=["text", "text"],
    title="Video to srt",
    description="Upload an audio file to Generate.",
)

iface.launch(inbrowser=True)
