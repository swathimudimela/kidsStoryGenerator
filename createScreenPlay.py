import os
import json
import re
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

def get_llm(model="gemini-3.6-flash", temperature=0.7):
    # Note: gemini-3.6-flash or gemini-2.5-pro / gemini-2.0-flash
    # Gemini models in google genai sdk
    try:
        return ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
        )
    except Exception as e:
        print(f"Error initializing LLM with {model}: {e}. Falling back to default.")
        return ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=temperature
        )

# Retain backward compatibility
llm = get_llm()

def load_transcript(file_path="transcription_result.txt") -> str:
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

def llm_segment_and_prompt(transcript: str, art_style: str = "colorful, 3D animated children's movie style", model: str = "gemini-3.6-flash"):
    custom_llm = get_llm(model=model)
    
    prompt = f"""
    You are a creative children's story editor and film director. 
    Analyze the following story transcript/idea and break it down into logical sequential scenes for a short animated storybook video.
    
    Style Guide for Visuals: {art_style}
    
    For each scene:
    1. Provide the exact story narration line (keep it fun, expressive, and engaging for kids).
    2. Provide a detailed image prompt suitable for {art_style}. Include key subjects, vibrant colors, expressive lighting, and scenery.

    Respond ONLY with a valid JSON array format matching this exact schema:
    [
      {{
        "narration": "Exact line to be spoken",
        "image_prompt": "Detailed description of the visual scene for image generation"
      }}
    ]

    Story Transcript:
    {transcript}
    """

    # Structured prompt call using standard LangChain invoke
    response = custom_llm.invoke(prompt)
    
    # Extract text cleanly whether content is a string or a list of blocks
    if isinstance(response.content, list):
        text_content = "".join(
            block["text"] if isinstance(block, dict) else getattr(block, "text", str(block))
            for block in response.content
        )
    else:
        text_content = str(response.content)

    # Strip markdown code blocks
    clean_text = text_content.strip()
    if "```" in clean_text:
        # Find json content inside backticks
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', clean_text)
        if match:
            clean_text = match.group(1).strip()
        else:
            clean_text = clean_text.replace("```json", "").replace("```", "").strip()
            
    try:
        data = json.loads(clean_text)
    except Exception as e:
        print(f"JSON parsing fallback: {e}")
        # Try finding the array inside brackets
        array_match = re.search(r'\[\s*\{.*\}\s*\]', clean_text, re.DOTALL)
        if array_match:
            data = json.loads(array_match.group(0))
        else:
            raise ValueError(f"Failed to parse scenes from LLM output: {clean_text[:200]}")
            
    return data if isinstance(data, list) else data.get("scenes", [])