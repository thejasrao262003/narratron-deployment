import random
from final_workflow.content_creation import generate_script, extract_json_from_markdown
from final_workflow.mongo_connection import insert_story_document, update_bg_music, get_scenes, push_image_file_names
from final_workflow.s3_operations import get_files, generate_presigned_url, upload_audio_file
from final_workflow.voice_generation import generate_audio
from final_workflow.combine_audio_image import create_video
from datetime import datetime
import logging

DEFAULT_IMAGE_KEY = "images/forest_trail_with_horror_mist_and_shadows/image_5.jpg"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)
today = datetime.today().date()

def run_workflow():
    # Step 1: Generate story
    response = generate_script()
    json_response = extract_json_from_markdown(response)
    insert_story_document(json_response)

    # Step 2: Pick background music
    file_keys = get_files("narratron", "bg_musics/")
    bg_musics = [key for key in file_keys if key.endswith(".mp3") or key.endswith(".wav")]
    bg_music_index = random.randint(0, len(bg_musics) - 1)
    bg_music_file = bg_musics[bg_music_index]
    update_bg_music(bg_music_file)

    # Step 3: Get scenes
    scenes = get_scenes()

    # Step 4: Generate audio for each scene
    audio_file_names = generate_audio(scenes)

    # Step 5: Pick and map image for each scene
    image_file_names = []
    for key, value in scenes.items():
        prefix = f"images/{value['scene_name'].replace(' ', '_').lower()}"
        image_keys = get_files("narratron", prefix)

        if not image_keys:
            logger.warning(f"⚠️ No images found in S3 for scene: {value['scene_name']}. Using default.")
            image_file_names.append(DEFAULT_IMAGE_KEY)
        else:
            image_file_names.append(random.choice(image_keys))

    push_image_file_names(image_file_names)

    # Step 6: Generate pre-signed URLs for assets
    img_presigned_urls = [generate_presigned_url(img) for img in image_file_names]
    audio_presigned_urls = [generate_presigned_url(audio) for audio in audio_file_names]
    bg_music_presigned_url = generate_presigned_url(bg_music_file)

    # Step 7: Render video
    create_video(img_presigned_urls, audio_presigned_urls, bg_music_presigned_url)

    # Step 8: Upload final video to S3
    with open("/tmp/FinalShorts/combined_shorts.mp4", "rb") as f:
        upload_audio_file(f, f"final_outputs/{today}/video_output.mp4", "narratron")