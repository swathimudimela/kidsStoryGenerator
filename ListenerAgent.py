import os
import base64
from pathlib import Path
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

def get_mime_type(suffix: str) -> str:
    s = suffix.lower()
    if s in [".mp3"]:
        return "audio/mp3"
    elif s in [".wav"]:
        return "audio/wav"
    elif s in [".m4a"]:
        return "audio/m4a"
    elif s in [".ogg"]:
        return "audio/ogg"
    elif s in [".webm"]:
        return "audio/webm"
    return "audio/mp3"

def transcribe_audio_langchain(audio_path: str, model: str = "gemini-3.6-flash") -> str:
    # 1. Verify file existence
    file_path = Path(audio_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found at: {audio_path}")

    # 2. Read and base64-encode the audio file
    print(f"Reading audio file: {audio_path}...")
    with open(file_path, "rb") as audio_file:
        encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")

    mime_type = get_mime_type(file_path.suffix)

    # 3. Initialize the Gemini chat model via LangChain
    try:
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=0.0
        )
    except Exception:
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            temperature=0.0
        )

    # 4. Define the multimodal prompt message
    prompt_text = """
    Listen to this audio carefully. It contains a child telling a story (possibly with an adult speaking in Telugu, English, or mixed languages).
    Provide a clean transcription including speaker labels. For non-English parts, provide the original spoken text and the English translation in parentheses.
    Make sure to capture the kid's story characters, dialogue, and fun expressions accurately.
    """

    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt_text},
            {
                "type": "media",
                "data": encoded_audio,
                "mime_type": mime_type
            }
        ]
    )

    # 5. Invoke the chain
    print("Processing audio with LangChain + Gemini...")
    response = llm.invoke([message])

    if isinstance(response.content, list):
        return "\n".join(
            doc.get("text", str(doc)) if isinstance(doc, dict) else (doc.page_content if hasattr(doc, 'page_content') else str(doc))
            for doc in response.content
        )
    return str(response.content)