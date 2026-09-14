import os
import requests
import soundfile as sf
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

output_dir = Path("generated_assets")
output_dir.mkdir(exist_ok=True)

_audio_pipeline = None

def get_audio_pipeline():
    global _audio_pipeline
    if _audio_pipeline is None:
        try:
            from kokoro import KPipeline
            # Explicitly pass repo_id to suppress the default repo warning
            _audio_pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')
        except Exception as e:
            print(f"Kokoro initialization warning: {e}. Will use gTTS fallback if needed.")
            _audio_pipeline = "gtts_fallback"
    return _audio_pipeline

def generate_single_audio(narration: str, audio_path: Path, voice: str = 'af_heart', speed: float = 1.0) -> bool:
    """Generates audio for a narration string using Kokoro or fallback to gTTS."""
    pipeline = get_audio_pipeline()
    if pipeline != "gtts_fallback":
        try:
            generator = pipeline(narration, voice=voice, speed=speed)
            for _, _, audio in generator:
                sf.write(str(audio_path), audio, 24000)
                return True
        except Exception as e:
            print(f"Kokoro audio generation failed for '{narration[:30]}...': {e}. Falling back to gTTS...")
    
    # Fallback to gTTS
    try:
        from gtts import gTTS
        tts = gTTS(text=narration, lang='en', slow=(speed < 0.9))
        tts.save(str(audio_path))
        return True
    except Exception as e:
        print(f"gTTS audio fallback also failed: {e}")
        return False

def generate_single_image(image_prompt: str, image_path: Path, art_style: str = "Pixar 3D render, vibrant, kid friendly") -> bool:
    """Generates an image via Pollinations.ai with art style prefix."""
    try:
        full_prompt = f"storybook style, {art_style}: {image_prompt}"
        encoded_prompt = requests.utils.quote(full_prompt)
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1280&height=720&nologo=true&seed={abs(hash(image_prompt)) % 100000}"
        
        response = requests.get(image_url, timeout=30)
        if response.status_code == 200:
            with open(image_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            print(f"Pollinations returned status {response.status_code}")
            return False
    except Exception as e:
        print(f"Image generation failed for prompt: {e}")
        return False

def generate_assets(scene_data, art_style="Pixar 3D render, vibrant, kid friendly", voice='af_heart', speed=1.0, progress_callback=None):
    processed_scenes = []
    total = len(scene_data)
    
    for idx, scene in enumerate(scene_data):
        scene_num = idx + 1
        msg = f"Processing Scene {scene_num}/{total}..."
        print(msg)
        if progress_callback:
            progress_callback(idx, total, f"🎨 Generating artwork & voiceover for Scene {scene_num}/{total}...")
        
        audio_path = output_dir / f"scene_{scene_num}.wav"
        image_path = output_dir / f"scene_{scene_num}.jpg"

        # 1. Audio Generation
        generate_single_audio(scene["narration"], audio_path, voice=voice, speed=speed)

        # 2. Image Generation
        generate_single_image(scene["image_prompt"], image_path, art_style=art_style)

        processed_scenes.append({
            "scene_number": scene_num,
            "narration": scene.get("narration", ""),
            "image_prompt": scene.get("image_prompt", ""),
            "image_path": str(image_path),
            "audio_path": str(audio_path)
        })
        
    if progress_callback:
        progress_callback(total, total, "✅ All scene assets generated!")
        
    return processed_scenes