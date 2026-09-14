import os
from pathlib import Path
from ListenerAgent import transcribe_audio_langchain
from createScreenPlay import load_transcript, llm_segment_and_prompt
from GenerateScenes import generate_assets
from assemble_video import assemble_video

if __name__ == "__main__":
    
    audio_file = "sample1.mp3"


    if os.path.exists(audio_file):
        transcript = transcribe_audio_langchain(audio_file)
        # Handle list output (strings or LangChain Document objects)
        if isinstance(transcript, list):
            # Extract page_content if elements are Document objects, else use the string as-is
            text_content = "\n".join(
                doc.page_content if hasattr(doc, 'page_content') else str(doc) 
                for doc in transcript
            )
        else:
            text_content = str(transcript)

        print("\n--- Transcription Result ---")
        print(text_content)

        # Write to a text file
        output_file_path = Path("transcription_result.txt")
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(text_content)
    else:
        print(f"File '{audio_file}' not found.")


    if os.path.exists("transcription_result.txt"):
        transcript = load_transcript()
        print("Parsing transcript with Gemini via LangChain...")
        scenes = llm_segment_and_prompt(transcript)
        
        print(f"LLM generated {len(scenes)} scenes.")
        asset_list = generate_assets(scenes)
        assemble_video(asset_list)
    else:
        print("transcription_result.txt not found. Please create it first.")