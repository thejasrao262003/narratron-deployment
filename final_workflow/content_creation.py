import google.generativeai as genai

import os
import sys
from dotenv import load_dotenv
import json
load_dotenv()

try:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set.")
    genai.configure(api_key=api_key)
except ValueError as e:
    print(f"Error: {e}")
    print("Please set the GOOGLE_API_KEY environment variable.")
    print("Get your key from: https://aistudio.google.com/app/apikey")
    sys.exit(1)


MODEL_NAME = 'gemini-2.0-flash'
model = genai.GenerativeModel(MODEL_NAME)

def generate_story_settings(max_tokens=1500):
    story_title_prompt = """
    You are an idea generator for horror youtube shorts.
    You don't need to generate a story. Just generate some variables that will be used to generate a story. 
    You need to ensure you don't repeat the same ideas.
    Output exampleformat:
    ```json
    {
        "story_title": "The horror doll that haunts the house",
        "time_period": "medieval",
        "setting": "abandoned village",
        "theme": "Annabelle",
    }
    ```
    """
    try:
        print("Sending request to Gemini API...")
        response = model.generate_content(story_title_prompt)
        print("Response received.")
        story_settings = extract_json_from_markdown(response.text)
        return story_settings
    except genai.types.generation_types.BlockedPromptException as e:
        print(f"Error: The prompt was blocked. Reason: {e}")
        return "Error: Prompt blocked by safety settings."
    except Exception as e:
        print(f"An error occurred during API communication: {e}")
        return None



def generate_script(max_tokens=1500):
    story_settings = generate_story_settings()
    prompt = f"""
    IMPORTANT: Strictly limit yourself to 6 scenes only. You are a hollywood script writer and an amazing content creator for horror scripts.
    I will be giving you a scene description and you need to write a script for it. 
    This is the story settings:
    {story_settings}
    Let title, description, tags and categoryId be youtube specific to more youtube views more.
    Write a story for a horror youtube short. Make it extremely scary. No sounds nothing. It should just be a story narrated by one narrator.
    I will give you a list of 166 scenes that I have images for. So you need to give me a json of. Each scene should have around 40-50 words of content. 
    End of 6th scene you need to have a suspense/cliff hanger ending that will make the viewers curious
    ```json
    {{
        story_name: "",
        title: "",
        description: "",
        tags: [],
        categoryId: 39
        scenes: {{
            scene1: {{
                scene_name: "",
                "scene_script": ""
            }},
            scene2: {{
                scene_name: "",
                "scene_script": ""
            }}...
        }}
    }}```
    Return only the JSON result as your final output.
    
    These are the scenes that I have images for:
        sunny suburban street with horror atmosphere
        abandoned village at sunset with eerie light
        haunted house with broken windows and fog
        old countryside house with eerie shadows
        forest trail with horror mist and shadows
        rustic wooden cabin deep in dark forest
        dinner table at dusk with creepy lighting
        figures walking into dense haunted forest
        garden with overgrown vines and sinister tone
        abandoned train track with horror setting
        car parked on foggy countryside road
        campfire in dark woods with spooky shadows
        empty road at sunset with haunting light
        person walking home on empty horror road
        forest clearing with disturbing atmosphere
        fog creeping through dense haunted woods
        shadows moving inside abandoned house
        tree with carved symbols in creepy forest
        cracked mirror with eerie reflection
        hallway with flickering light horror style
        abandoned school hallway with decayed walls
        black cat staring in a horror setting
        wooden staircase in haunted mansion
        rocking chair with old doll in shadow
        open forest door leading into darkness
        dusty window with bloody handprint
        soundproof basement with dim light horror
        unmade bed in abandoned horror room
        haunted forest at night with red mist
        silhouette behind old curtain horror style
        levitating person in horror environment
        knife on sink with blood horror setting
        shadow in dark room corner horror
        elderly woman with black eyes horror face
        abandoned hallway with peeling walls
        graveyard scene with hands from ground
        mirror cracked with blood splatter horror
        figure behind shower curtain horror scene
        wall with demonic symbol in horror tone
        dark hospital corridor horror theme
        ghost in white dress floating in room
        room filled with antique creepy dolls
        bloody footprints across wooden floor
        exterior of abandoned asylum horror
        night view of haunted Victorian mansion
        attic with cobwebs and single lightbulb
        music box with broken parts horror vibe
        creepy painting with glowing red eyes
        clock striking midnight in empty house
        cemetery with thick fog and moonlight
        stained glass window cracked horror style
        rocking chair in moonlit attic horror
        wall mirror in haunted horror room
        underground crypt with eerie lanterns
        rundown motel with broken neon lights
        amusement park abandoned at night
        graveyard statue with cracked face
        broken ride in abandoned carnival
        close-up eyes filled with fear horror
        person screaming in pitch black room
        figure hiding under old bed horror
        person with anxious expression horror
        man hiding face in horror setting
        hands holding candle in complete darkness
        figure crying in corner of horror room
        shadow behind curtain in haunted room
        clenched fists with blood horror detail
        person peeking through keyhole creepy
        mouth whispering close-up horror feel
        silhouette running in fear horror
        half-lit face in creepy setting
        mouth stitched shut horror concept
        zombie face with decaying flesh
        ghostly figure with hollow eyes
        demonic clown face in shadows
        witch with glowing green eyes
        shadow creature crawling in hallway
        vampire with blood around mouth
        possessed man in dark chained room
        headless figure standing in fog
        monster behind old wooden door
        tall silhouette in forest mist
        creature with long twisted limbs
        ghoul rising from black water
        scarecrow with blood stains
        woman without face in black gown
        hooded figure with lantern at night
        ghost captured on old camera footage
        hallway with flickering horror light
        photo with ghostly figure in background
        camera flash in abandoned room horror
        horror setup with EVP recorder
        shadow figure in security footage
        ghost in night vision footage
        camera above empty crib horror setup
        ghost handprint on camera lens
        broken camera lens after ghost photo
        burned photo album from past horror
        storm clouds over abandoned house
        broken window with moonlight shadow
        headlights through fog on road
        tree scratching window horror style
        rain dripping on broken glass window
        strong wind blowing forest branches
        lightning above creepy mansion
        dense fog covering old cemetery
        rain falling on abandoned street
        storm leaves blowing in horror tone
        solar eclipse over dead town
        candle casting long horror shadow
        moonlight through forest branches
        tree bending from ghostly force
        wind rattling broken windows
        burned ruins of house in daylight
        lone figure walking horror road
        sunrise behind graveyard horror tone
        mystery house with police tape
        hospital bed with horror IV setup
        clouds parting over ruined town
        smoke rising from dark forest
        survivor sitting alone on porch
        hand holding faded photo horror memory
        survivor walking into burning sky
        blood stain being washed by rain
        horror note on door in ruins
        sunlight over destroyed buildings
        tunnel entrance into darkness
        flashlight in pitch black corridor
        door creaking open slowly
        shadows moving on wall horror
        ceiling fan spinning in silence
        hand turning doorknob slowly
        rocking horse in dark corner
        ticking clock at horror midnight
        lightbulb bursting in silence
        static on TV screen in dark room
        long shadow cast on wall horror
        upside down cross on wall
        blood dripping from ceiling
        candle going out in storm
        glass breaking in slow motion
        fog covering entire haunted road
        scream echoing from old tunnel
        bloody writing on mirror horror
        ghost woman in tattered dress standing by old well
        Ghostly figure hovering in foggy graveyard at night
        Ghost apparition gliding through abandoned hallway
        Pale ghost with hollow eyes emerging from misty forest
        ghost standing silently in deserted playground
        horror silhouette lurking behind cracked window at dusk
        horror figure standing under flickering streetlamp in alley
        horror presence casting long shadow in empty corridor
        Faceless ghost watching from behind tree in dense woods
        Demon with glowing red eyes crawling on ceiling
        Possessed ghost levitating above bed horror
        Twisted face in mirror with blood smears horror
        Horned demon emerging from circle of fire in basement without watermarks
        Old rocking chair with ghost doll moving on its own in dusty attic
        Blood-stained horror handprints leading down dark staircase
        Ancient portrait horror
    """
    try:
        print("Sending request to Gemini API...")
        response = model.generate_content(prompt)
        print("Response received.")
        return response.text
    except genai.types.generation_types.BlockedPromptException as e:
        print(f"Error: The prompt was blocked. Reason: {e}")
        return "Error: Prompt blocked by safety settings."
    except Exception as e:
        print(f"An error occurred during API communication: {e}")
        return None

def extract_json_from_markdown(text: str) -> dict:
    if text.strip().startswith("```json"):
        text = text.strip().removeprefix("```json").removesuffix("```").strip()
    elif text.strip().startswith("```"):
        text = text.strip().removeprefix("```").removesuffix("```").strip()

    return json.loads(text)


