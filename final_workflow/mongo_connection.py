from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from datetime import datetime, timedelta
from bson import ObjectId
from dotenv import load_dotenv
import os
load_dotenv()

# --- MongoDB Connection Setup ---
uri = os.getenv("MONGO_URI")
client = MongoClient(uri, server_api=ServerApi('1'))
today = datetime.today().date()
start = datetime.combine(today, datetime.min.time())
end = datetime.combine(today + timedelta(days=1), datetime.min.time())
try:
    client.admin.command('ping')
    print("✅ Connected to MongoDB")
except Exception as e:
    print("❌ Connection failed:", e)
    exit()

db = client["Story"]
scenes_collection = db["scenes"]


def insert_story_document(response: dict):
    try:
        story_title = response.get("title", "")
        description = response.get("description", "")
        tags = response.get("tags", [])
        category = response.get("categoryId", 0)
        scene_data = response.get("scenes", {})

        story_doc = {
            "_id": ObjectId(),
            "created_date": datetime.today(),
            "title": story_title,
            "description": description,
            "keywords": ", ".join(tags),
            "category": category,
            "bg_music": "",
            "file": "",
            "scenes": scene_data
        }

        result = scenes_collection.insert_one(story_doc)
        print(f"✅ Inserted full story with ID: {result.inserted_id}")
    except Exception as e:
        print("❌ Failed to insert story:", e)


def update_bg_music(bg_music_name: str):
    try:
        result = scenes_collection.update_one(
            {"created_date": {"$gte": start, "$lt": end}},
            {"$set": {"bg_music": bg_music_name}}
        )

        if result.modified_count > 0:
            print(f"✅ Updated bg_music to '{bg_music_name}' for document on {today}")
        else:
            print(f"⚠️ No document found with created_date on {today} to update.")
    except Exception as e:
        print("❌ Failed to update bg_music:", e)


def get_scenes():
    try:
        document = scenes_collection.find_one({"created_date": {"$gte": start, "$lt": end}})

        if document:
            print("✅ Found today's document")
            return document.get("scenes")
        else:
            print("⚠️ No scene document found for today.")
            return None
    except Exception as e:
        print("❌ Error while fetching scenes:", e)
        return None


def push_audio_file_names(file_names: list):
    try:
        document = scenes_collection.find_one({"created_date": {"$gte": start, "$lt": end}})

        if not document or "scenes" not in document:
            print("❌ No valid document with 'scenes' found for today.")
            return

        scenes = document["scenes"]

        if not isinstance(scenes, dict):
            print("❌ 'scenes' field is not a dict. Skipping update.")
            return

        updated_scenes = {}
        for i, (key, scene) in enumerate(scenes.items()):
            scene["audio_file_name"] = file_names[i] if i < len(file_names) else None
            updated_scenes[key] = scene

        scenes_collection.update_one(
            {"_id": document["_id"]},
            {"$set": {"scenes": updated_scenes}}
        )
        print(f"✅ Updated scenes with audio file names.")

    except Exception as e:
        print("❌ Failed to update scenes with audio file names:", e)

def push_image_file_names(file_names: list):
    try:
        document = scenes_collection.find_one({"created_date": {"$gte": start, "$lt": end}})
        if not document or "scenes" not in document:
            print("❌ No valid document with 'scenes' found for today.")
            return
        scenes = document["scenes"]
        if not isinstance(scenes, dict):
            print("❌ 'scenes' field is not a dict. Skipping update.")
            return
        updated_scenes = {}
        for i, (key, scene) in enumerate(scenes.items()):
            scene["image_file_name"] = file_names[i] if i < len(file_names) else None
            updated_scenes[key] = scene
        scenes_collection.update_one(
            {"_id": document["_id"]},
            {"$set": {"scenes": updated_scenes}}
        )
        print(f"✅ Updated scenes with image file names.")

    except Exception as e:
        print("❌ Failed to update scenes with image file names:", e)

