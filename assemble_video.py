from pathlib import Path
import os
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

def assemble_video(scenes, output_filename="kids_story_movie.mp4", progress_callback=None):
    if progress_callback:
        progress_callback(0.2, "🎞️ Loading audio and visual clips...")
        
    clips = []
    for scene in scenes:
        img_path = str(scene["image_path"])
        aud_path = str(scene["audio_path"])
        
        if os.path.exists(img_path) and os.path.exists(aud_path):
            audio_clip = AudioFileClip(aud_path)
            if audio_clip.duration and audio_clip.duration > 0:
                image_clip = (
                    ImageClip(img_path)
                    .with_duration(audio_clip.duration)
                    .with_audio(audio_clip)
                )
                clips.append(image_clip)
            else:
                print(f"Warning: Zero duration for {aud_path}")
    
    if not clips:
        print("No valid clips to assemble.")
        return None

    if progress_callback:
        progress_callback(0.5, f"🎬 Stitching {len(clips)} scenes into final movie...")
        
    print(f"\n--- Compiling Final Video ({len(clips)} clips) ---")
    final_video = concatenate_videoclips(clips, method="compose")
    
    output_path = Path(output_filename).resolve()
    temp_audio = str(output_path.parent / "temp-audio.m4a")
    
    final_video.write_videofile(
        str(output_path),
        fps=24,
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=temp_audio,
        remove_temp=True
    )
    
    # Safely close clip handles on Windows
    for clip in clips:
        try:
            clip.close()
            if clip.audio:
                clip.audio.close()
        except Exception:
            pass
    try:
        final_video.close()
    except Exception:
        pass
        
    if progress_callback:
        progress_callback(1.0, "🎉 Video rendered successfully!")
        
    print(f"Success! Output saved to: {output_path}")
    return str(output_path)