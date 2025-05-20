import os
import requests
import whisper
from io import BytesIO
from typing import List, Tuple
from pydub import AudioSegment
from moviepy.editor import (
    ImageClip,
    AudioFileClip,
    TextClip,
    CompositeVideoClip
)
from moviepy.video.fx.all import crop

TMP_DIR = "/tmp"
OUTPUT_FOLDER = os.path.join(TMP_DIR, "FinalShorts")
FINAL_VIDEO_PATH = os.path.join(OUTPUT_FOLDER, "combined_shorts.mp4")

def download_file_from_url(url: str) -> BytesIO:
    response = requests.get(url)
    if response.status_code == 200:
        data = BytesIO(response.content)
        data.seek(0)
        return data
    else:
        raise Exception(f"Failed to download from {url}")

# Slow down audio without changing pitch drastically
def time_stretch_audio(audio: AudioSegment, rate: float) -> AudioSegment:
    new_frame_rate = int(audio.frame_rate * rate)
    stretched = audio._spawn(audio.raw_data, overrides={'frame_rate': new_frame_rate})
    return stretched.set_frame_rate(audio.frame_rate)

def mix_audio_with_bgm(main_audio: AudioSegment, bgm: AudioSegment) -> AudioSegment:
    bgm_loop = bgm * (len(main_audio) // len(bgm) + 1)
    return main_audio.overlay(bgm_loop[:len(main_audio)])

def combine_all_audio(audio_files: List[BytesIO], bgm: AudioSegment, stretch_factor: float = 0.95) -> Tuple[str, List[float]]:
    combined = AudioSegment.empty()
    durations = []

    for file_io in audio_files:
        original_audio = AudioSegment.from_file(file_io)
        slowed_audio = time_stretch_audio(original_audio, stretch_factor)
        mixed = mix_audio_with_bgm(slowed_audio, bgm)
        durations.append(len(mixed) / 1000)
        combined += mixed

    output_path = os.path.join(TMP_DIR, "temp_full_audio.wav")
    combined.export(output_path, format="wav")
    return output_path, durations

def transcribe_audio(path: str, model) -> List[dict]:
    result = model.transcribe(path, word_timestamps=False)
    return result["segments"]

def compute_image_switch_points(durations: List[float]) -> List[float]:
    switch_times = []
    cum_time = 0
    for dur in durations:
        cum_time += dur
        switch_times.append(cum_time)
    return switch_times

def get_image_for_time(t: float, switch_times: List[float], img_paths: List[str]) -> str:
    for i, switch in enumerate(switch_times):
        if t < switch:
            return img_paths[i % len(img_paths)]
    return img_paths[-1]

def create_video_and_subtitle_clips(segments: List[dict], switch_times: List[float], img_paths: List[str]) -> Tuple[List[ImageClip], List[TextClip]]:
    video_clips = []
    subtitle_clips = []

    for seg in segments:
        text = seg['text'].strip()
        if not text:
            continue
        start, end = seg['start'], seg['end']
        image_path = get_image_for_time(start, switch_times, img_paths)

        image_clip = (
            ImageClip(image_path)
            .set_start(start)
            .set_duration(end - start)
            .resize(height=1920)
        )
        image_clip = crop(image_clip, width=1080, height=1920, x_center=image_clip.w / 2)

        txt_clip = (
            TextClip(text, fontsize=50, color='white', font='Arial-Bold', method='caption', size=(1000, None))
            .set_position(("center", 1600))
            .set_start(start)
            .set_duration(end - start)
        )

        video_clips.append(image_clip)
        subtitle_clips.append(txt_clip)

    return video_clips, subtitle_clips

def create_final_video(video_clips: List[ImageClip], subtitle_clips: List[TextClip], audio_path: str, output_path: str):
    composite = CompositeVideoClip(video_clips + subtitle_clips)
    final_audio = AudioFileClip(audio_path)
    final_video = composite.set_audio(final_audio)
    final_video.write_videofile(output_path, fps=30, codec="libx264", audio_codec="aac")

def create_video(img_list: List[str], audio_list: List[str], bg_music: str):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    image_paths = []
    audio_files = []

    # Download image files
    for idx, img_url in enumerate(img_list):
        img_data = download_file_from_url(img_url)
        img_path = os.path.join(TMP_DIR, f"temp_image_{idx}.jpg")
        with open(img_path, "wb") as f:
            f.write(img_data.read())
        image_paths.append(img_path)

    # Download audio files
    for audio_url in audio_list:
        audio_files.append(download_file_from_url(audio_url))

    # Download and prepare background music
    bg_music_data = download_file_from_url(bg_music)
    bg_music_segment = AudioSegment.from_file(bg_music_data)

    # Combine & slow audio, mix with BGM
    combined_audio_path, durations = combine_all_audio(audio_files, bg_music_segment, stretch_factor=0.95)
    switch_times = compute_image_switch_points(durations)

    # Transcribe slowed audio
    model = whisper.load_model("base")
    segments = transcribe_audio(combined_audio_path, model)

    # Generate video + captions
    video_clips, subtitle_clips = create_video_and_subtitle_clips(segments, switch_times, image_paths)

    # Final assembly
    create_final_video(video_clips, subtitle_clips, combined_audio_path, FINAL_VIDEO_PATH)

    # Clean up
    os.remove(combined_audio_path)
    for path in image_paths:
        os.remove(path)
