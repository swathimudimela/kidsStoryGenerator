import os
import time
import json
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ListenerAgent import transcribe_audio_langchain
from createScreenPlay import llm_segment_and_prompt
from GenerateScenes import generate_assets, generate_single_image, generate_single_audio
from assemble_video import assemble_video

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Kids Movie Maker Studio",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (Kid-Friendly, Modern, Playful)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=Quicksand:wght@500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', 'Quicksand', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 40%, #FFA07A 70%, #9B51E0 100%);
        padding: 26px 32px;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(255, 107, 107, 0.25);
    }
    
    .main-header h1 {
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.4rem;
        margin: 0;
        letter-spacing: -0.5px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.15);
    }
    
    .main-header p {
        font-size: 1.1rem;
        margin-top: 8px;
        opacity: 0.95;
        font-weight: 500;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 16px;
        padding: 20px;
        margin-bottom: 20px;
        backdrop-filter: blur(10px);
    }

    .scene-card {
        background: #1e1e2f;
        border: 2px solid #3d3b54;
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .scene-card:hover {
        border-color: #FF8E53;
        transform: translateY(-2px);
    }

    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .badge-scene {
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        color: white;
    }

    .step-number {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        background: #FF8E53;
        color: white;
        border-radius: 50%;
        font-weight: bold;
        margin-right: 10px;
    }
    
    .stButton>button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# App State Initialization
# -----------------------------------------------------------------------------
if "transcript" not in st.session_state:
    st.session_state.transcript = ""
if "scenes" not in st.session_state:
    st.session_state.scenes = []
if "asset_list" not in st.session_state:
    st.session_state.asset_list = []
if "final_video_path" not in st.session_state:
    st.session_state.final_video_path = None
if "active_audio_path" not in st.session_state:
    st.session_state.active_audio_path = None

output_dir = Path("generated_assets")
output_dir.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Sidebar: Settings & Magic Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🎨 Visual & Voice Studio")
    
    # Visual Art Style Selection
    style_presets = {
        "✨ Pixar 3D Animation": "storybook style, Pixar 3D render, vibrant, kid friendly, cute volumetric lighting",
        "🧸 Claymation / Stop-Motion": "claymation, handcrafted plasticine, stop motion animation style, tactile textures",
        "🎨 Storybook Watercolor": "whimsical watercolor illustration, children's storybook art, soft vibrant pastel palette",
        "🌟 Studio Ghibli Anime": "Studio Ghibli anime aesthetic, lush hand-painted background, magical atmosphere",
        "🖍️ 2D Vibrant Cartoon": "cute 2D children cartoon style, bold clean outlines, cheerful bright colors",
        "🚀 Retro Sci-Fi / Fantasy": "retro futuristic storybook art, glowing colors, whimsical fantasy details"
    }
    
    selected_style_name = st.selectbox(
        "Visual Art Style",
        options=list(style_presets.keys()),
        index=0
    )
    art_style_prompt = style_presets[selected_style_name]
    
    custom_style = st.text_input("Customize Style Prompt (Optional)", value="")
    if custom_style.strip():
        art_style_prompt = custom_style.strip()

    st.markdown("---")
    st.markdown("### 🎙️ Voiceover Settings")
    
    voice_options = {
        "af_heart (Warm & Gentle Female - Kokoro)": "af_heart",
        "af_bella (Enthusiastic Female - Kokoro)": "af_bella",
        "af_nicole (Friendly Female - Kokoro)": "af_nicole",
        "am_adam (Warm Narrator Male - Kokoro)": "am_adam",
        "am_michael (Storyteller Male - Kokoro)": "am_michael",
        "bf_alice (Storybook British Female)": "bf_alice"
    }
    selected_voice_label = st.selectbox("Narrator Voice", options=list(voice_options.keys()), index=0)
    voice_code = voice_options[selected_voice_label]

    voice_speed = st.slider("Speech Speed", min_value=0.8, max_value=1.3, value=1.0, step=0.05)

    st.markdown("---")
    st.markdown("### ⚙️ AI Models")
    gemini_model = st.selectbox("Gemini Model", options=["gemini-3.6-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro"], index=0)

    st.markdown("---")
    if st.button("🧹 Clear All & Start Fresh", use_container_width=True):
        st.session_state.transcript = ""
        st.session_state.scenes = []
        st.session_state.asset_list = []
        st.session_state.final_video_path = None
        st.session_state.active_audio_path = None
        st.rerun()

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🎬 Kids Story Movie Maker Studio 🍿</h1>
    <p>Turn children's voice recordings & imaginative ideas into animated video movies with AI Magic!</p>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 1: Story Input
# -----------------------------------------------------------------------------
st.markdown("### 1️⃣ Provide the Child's Story")
input_tab1, input_tab2, input_tab3, input_tab4 = st.tabs([
    "🎵 Upload Audio File", 
    "🎙️ Record Audio Now", 
    "⭐ Load Sample Story (sample1.mp3)", 
    "✍️ Type / Paste Story Text"
])

# Tab 1: Upload Audio
with input_tab1:
    uploaded_file = st.file_uploader(
        "Upload audio recording (Telugu, English, or any language)",
        type=["mp3", "wav", "m4a", "ogg"]
    )
    if uploaded_file is not None:
        save_path = Path("temp_uploaded_audio" + Path(uploaded_file.name).suffix)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state.active_audio_path = str(save_path)
        st.audio(str(save_path))
        st.success(f"Loaded: `{uploaded_file.name}`")

# Tab 2: Record Audio
with input_tab2:
    recorded_audio = st.audio_input("Record child speaking into microphone:")
    if recorded_audio is not None:
        save_path = Path("recorded_story.wav")
        with open(save_path, "wb") as f:
            f.write(recorded_audio.getbuffer())
        st.session_state.active_audio_path = str(save_path)
        st.audio(str(save_path))
        st.success("Audio recorded successfully!")

# Tab 3: Sample Story
with input_tab3:
    col_sample1, col_sample2 = st.columns([1, 2])
    with col_sample1:
        if st.button("✨ Load `sample1.mp3`", use_container_width=True):
            if os.path.exists("sample1.mp3"):
                st.session_state.active_audio_path = "sample1.mp3"
                st.success("Loaded `sample1.mp3`!")
            else:
                st.error("`sample1.mp3` was not found in project directory.")
    with col_sample2:
        if os.path.exists("sample1.mp3"):
            st.audio("sample1.mp3")
            st.caption("Included Telugu-English child conversation sample.")

# Tab 4: Direct Story Text
with input_tab4:
    direct_story_text = st.text_area(
        "Direct Story Idea / Script (Bypass Audio Transcription)",
        placeholder="Once upon a time, a brave little puppy found a magical red rocket in the backyard...",
        height=120
    )
    if st.button("📝 Use This Story Text", use_container_width=True):
        if direct_story_text.strip():
            st.session_state.transcript = direct_story_text.strip()
            st.session_state.active_audio_path = None
            st.success("Story transcript set from text input!")
        else:
            st.warning("Please enter some text first.")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Section 2: Magic One-Click vs Step-by-Step
# -----------------------------------------------------------------------------
col_magic, col_info = st.columns([1, 2])

with col_magic:
    magic_btn = st.button("🚀 Generate Full Movie (1-Click Magic)", type="primary", use_container_width=True)

with col_info:
    st.info("💡 You can run the entire pipeline automatically with **1-Click Magic**, or use the step-by-step sections below to review and edit each stage.")

# =============================================================================
# 1-CLICK MAGIC PIPELINE EXECUTION
# =============================================================================
if magic_btn:
    progress_bar = st.progress(0, text="Starting Magic Movie Pipeline...")
    
    # 1. Transcription (if audio available)
    if st.session_state.active_audio_path:
        progress_bar.progress(15, text="🎙️ Step 1/4: Transcribing kid's audio with Gemini...")
        try:
            transcript = transcribe_audio_langchain(st.session_state.active_audio_path, model=gemini_model)
            st.session_state.transcript = transcript
            with open("transcription_result.txt", "w", encoding="utf-8") as f:
                f.write(transcript)
        except Exception as e:
            st.error(f"Transcription error: {e}")
            st.stop()
    elif not st.session_state.transcript:
        st.warning("Please provide an audio file or enter story text above!")
        st.stop()
        
    # 2. Screenplay breakdown
    progress_bar.progress(35, text="🎬 Step 2/4: Director Agent generating screenplay scenes...")
    try:
        scenes = llm_segment_and_prompt(st.session_state.transcript, art_style=art_style_prompt, model=gemini_model)
        st.session_state.scenes = scenes
    except Exception as e:
        st.error(f"Screenplay generation error: {e}")
        st.stop()
        
    # 3. Generate Assets
    def asset_progress(cur, total, msg):
        pct = 35 + int((cur / max(total, 1)) * 40)
        progress_bar.progress(pct, text=f"🎨 Step 3/4: {msg}")
        
    try:
        asset_list = generate_assets(
            st.session_state.scenes,
            art_style=art_style_prompt,
            voice=voice_code,
            speed=voice_speed,
            progress_callback=asset_progress
        )
        st.session_state.asset_list = asset_list
    except Exception as e:
        st.error(f"Asset generation error: {e}")
        st.stop()
        
    # 4. Assemble Video
    progress_bar.progress(85, text="🍿 Step 4/4: Stitching scenes into final animated movie...")
    try:
        def assemble_prog(pct_val, msg):
            progress_bar.progress(85 + int(pct_val * 15), text=f"🍿 {msg}")
            
        video_path = assemble_video(
            st.session_state.asset_list,
            output_filename="kids_story_movie.mp4",
            progress_callback=assemble_prog
        )
        st.session_state.final_video_path = video_path
        progress_bar.progress(100, text="🎉 Magic Movie Created Successfully!")
        st.balloons()
    except Exception as e:
        st.error(f"Video assembly error: {e}")
        st.stop()

# =============================================================================
# STEP-BY-STEP WORKFLOW ACCORDION
# =============================================================================
st.markdown("---")
st.markdown("### 🛠️ Interactive Creative Studio")

step1_expander = st.expander("🎙️ Step 1: Listener Agent (Transcription)", expanded=bool(st.session_state.transcript))
with step1_expander:
    st.caption("Converts raw bilingual/child speech into a clean structured story transcript.")
    col_t1, col_t2 = st.columns([1, 3])
    with col_t1:
        if st.button("🎧 Run Transcription", key="btn_transcribe", use_container_width=True):
            if st.session_state.active_audio_path:
                with st.spinner("Transcribing audio with Gemini..."):
                    try:
                        trans = transcribe_audio_langchain(st.session_state.active_audio_path, model=gemini_model)
                        st.session_state.transcript = trans
                        with open("transcription_result.txt", "w", encoding="utf-8") as f:
                            f.write(trans)
                        st.success("Transcription complete!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
            else:
                st.warning("Please upload, record, or select sample audio in Section 1 first.")
    with col_t2:
        edited_transcript = st.text_area(
            "Review & Edit Transcript:",
            value=st.session_state.transcript,
            height=140,
            key="transcript_editor"
        )
        if edited_transcript != st.session_state.transcript:
            st.session_state.transcript = edited_transcript

step2_expander = st.expander("🎬 Step 2: Director Agent (Screenplay & Scene Breakdown)", expanded=bool(st.session_state.scenes))
with step2_expander:
    st.caption("Breaks down the story into visual scenes with exact narration and art prompts.")
    col_s1, col_s2 = st.columns([1, 3])
    with col_s1:
        if st.button("📝 Generate Screenplay", key="btn_screenplay", use_container_width=True):
            if st.session_state.transcript:
                with st.spinner("Director Agent drafting scenes..."):
                    try:
                        scenes = llm_segment_and_prompt(st.session_state.transcript, art_style=art_style_prompt, model=gemini_model)
                        st.session_state.scenes = scenes
                        st.success(f"Generated {len(scenes)} scenes!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Screenplay creation failed: {e}")
            else:
                st.warning("Please provide or transcribe a story transcript in Step 1 first.")
    
    with col_s2:
        if st.session_state.scenes:
            st.markdown(f"**Found {len(st.session_state.scenes)} Story Scenes:**")
            for idx, sc in enumerate(st.session_state.scenes):
                with st.container():
                    st.markdown(f"**Scene {idx + 1}**")
                    col_sc_narr, col_sc_prompt = st.columns([1, 1])
                    with col_sc_narr:
                        new_narr = st.text_area(
                            f"Narration line (Scene {idx+1}):",
                            value=sc.get("narration", ""),
                            key=f"sc_narr_{idx}",
                            height=80
                        )
                        sc["narration"] = new_narr
                    with col_sc_prompt:
                        new_prompt = st.text_area(
                            f"Visual Image Prompt (Scene {idx+1}):",
                            value=sc.get("image_prompt", ""),
                            key=f"sc_prompt_{idx}",
                            height=80
                        )
                        sc["image_prompt"] = new_prompt
                    st.divider()

step3_expander = st.expander("🎨 Step 3: Media Production (Generate Voiceover & Art)", expanded=bool(st.session_state.asset_list))
with step3_expander:
    st.caption("Synthesizes Kokoro voiceover audio and Pollinations 3D visuals for every scene.")
    if st.button("🎨 Generate All Scene Assets", key="btn_gen_assets"):
        if st.session_state.scenes:
            prog = st.progress(0, text="Generating assets...")
            def ui_callback(cur, tot, msg):
                prog.progress(int((cur / max(tot, 1)) * 100), text=msg)
                
            with st.spinner("Generating artwork & speech..."):
                try:
                    assets = generate_assets(
                        st.session_state.scenes,
                        art_style=art_style_prompt,
                        voice=voice_code,
                        speed=voice_speed,
                        progress_callback=ui_callback
                    )
                    st.session_state.asset_list = assets
                    st.success("All assets generated!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Asset generation failed: {e}")
        else:
            st.warning("Please generate scenes in Step 2 first.")
            
    # Storyboard display
    if st.session_state.asset_list:
        st.markdown("#### 🖼️ Storyboard Preview")
        cols = st.columns(min(len(st.session_state.asset_list), 3))
        for idx, item in enumerate(st.session_state.asset_list):
            with cols[idx % 3]:
                st.markdown(f"<span class='badge-pill badge-scene'>Scene {idx + 1}</span>", unsafe_allow_html=True)
                if os.path.exists(item["image_path"]):
                    st.image(item["image_path"], use_container_width=True)
                if os.path.exists(item["audio_path"]):
                    st.audio(item["audio_path"])
                st.caption(f"**Narration:** {item.get('narration', '')}")
                
                # Per-scene regeneration
                if st.button(f"🔄 Re-draw Scene {idx + 1}", key=f"regen_img_{idx}"):
                    with st.spinner(f"Re-generating Image {idx+1}..."):
                        img_p = Path(item["image_path"])
                        # Add small randomizer to prompt
                        rand_prompt = item.get("image_prompt", "") + f" variation {int(time.time()) % 100}"
                        generate_single_image(rand_prompt, img_p, art_style=art_style_prompt)
                        st.rerun()

step4_expander = st.expander("🍿 Step 4: Assemble Final Video", expanded=bool(st.session_state.final_video_path))
with step4_expander:
    st.caption("Merges visual storybook stills with voiceover tracks into a synchronized MP4 movie.")
    if st.button("🎬 Assemble Final Movie Now", key="btn_assemble", type="primary"):
        if st.session_state.asset_list:
            with st.spinner("Stitching video with MoviePy..."):
                try:
                    out_vid = assemble_video(st.session_state.asset_list, output_filename="kids_story_movie.mp4")
                    st.session_state.final_video_path = out_vid
                    st.success("Final video assembled successfully!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Video assembly failed: {e}")
        else:
            st.warning("Please generate scene assets in Step 3 first.")

# =============================================================================
# FINAL MOVIE THEATER & EXPORT
# =============================================================================
if st.session_state.final_video_path and os.path.exists(st.session_state.final_video_path):
    st.markdown("---")
    st.markdown("## 🍿 ✨ Final Movie Theater ✨")
    
    col_theater1, col_theater2 = st.columns([3, 2])
    
    with col_theater1:
        st.video(st.session_state.final_video_path)
        
    with col_theater2:
        st.markdown("### 🏆 Movie Information")
        file_size_mb = os.path.getsize(st.session_state.final_video_path) / (1024 * 1024)
        st.markdown(f"- **Scene Count**: `{len(st.session_state.scenes)} Scenes`")
        st.markdown(f"- **Visual Art Style**: `{selected_style_name}`")
        st.markdown(f"- **Narrator Voice**: `{voice_code}`")
        st.markdown(f"- **File Size**: `{file_size_mb:.2f} MB`")
        st.markdown(f"- **Resolution**: `1280x720 (HD 16:9)`")
        
        with open(st.session_state.final_video_path, "rb") as vid_file:
            st.download_button(
                label="📥 Download Movie (.mp4)",
                data=vid_file,
                file_name="kids_story_movie.mp4",
                mime="video/mp4",
                type="primary",
                use_container_width=True
            )
