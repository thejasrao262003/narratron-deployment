import requests
from final_workflow.s3_operations import upload_audio_file
from final_workflow.mongo_connection import push_audio_file_names
from io import BytesIO
from datetime import datetime

today = datetime.today().date()
BUCKET_NAME = "narratron"

def generate_audio(scenes, voice="Alexander"):
    files_names = []

    for i, (key, value) in enumerate(scenes.items()):
        try:
            scene_name = value.get('scene_name', f"unnamed_{i}").replace(' ', '_').lower()
            scene_text = value.get('scene_script', '')
            scene_text += " . . ."

            print(f"[DEBUG] Generating audio for scene {i + 1}: {scene_name}")

            response = requests.post(
                "https://thejasrao262003--dia-tts-service-serve.modal.run/simple-tts",
                json={
                    "text": scene_text,
                    "speed": 0.94,
                    "seed": 42,
                    "voice": voice
                }
            )

            if response.status_code != 200:
                print(f"[ERROR] Failed to generate audio for scene '{scene_name}': {response.text}")
                continue
#25, 66, 97, 98, 120, 126
            audio_buffer = BytesIO(response.content)
            audio_buffer.seek(0)

            filename = f"scene_{scene_name}.wav"
            s3_path = f"text_to_speech/{today}/{filename}"

            try:
                upload_audio_file(audio_buffer, s3_path, BUCKET_NAME)
                print(f"[DEBUG] Uploaded: {s3_path}")
                files_names.append(s3_path)
            except Exception as upload_err:
                print(f"[ERROR] Failed to upload {filename} to S3: {upload_err}")
                continue

        except Exception as e:
            print(f"[ERROR] Unexpected error while processing scene {i + 1}: {e}")
            continue

    try:
        print("[DEBUG] Updating MongoDB with audio file names...")
        push_audio_file_names(files_names)
        print("[DEBUG] MongoDB update successful.")
        return files_names
    except Exception as mongo_err:
        print(f"[ERROR] Failed to update MongoDB: {mongo_err}")
        return None
