import streamlit as st
from groq import Groq
import difflib
from newspaper import Article
import pandas as pd
import plotly.express as px
from datetime import datetime
from typing import List, Dict, Tuple
import time
import os
import json
import re
from dotenv import load_dotenv


GROQ_API_KEY = "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"

# --- SETUP --- #
st.set_page_config(
    page_title="📰 Objective News Translator (Llama 3.3 70B)",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Groq client
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

# --- CORE FUNCTIONS (Llama 3.3 70B Enhanced) --- #
def analyze_with_llama(system_prompt: str, user_input: str, max_tokens: int = 4000) -> str:
    """Query Llama 3.3 70B via Groq's ultra-fast API."""
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",  # Updated model name
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input[:max_tokens]}  # Truncate to fit context
            ],
            temperature=0.3,  # Lower = more factual
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        return ""

def detect_biases(text: str) -> Dict:
    """Advanced bias detection with explanations."""
    PROMPT = """Analyze this text for:
    1. Loaded language (emotional/subjective words)
    2. Omitted context (what's missing?)
    3. Political slant (Left/Center/Right)
    4. Suggested neutral rewrites
    
    Return as JSON with keys: loaded_words, missing_context, slant_score (1-10), neutral_rewrites"""
    result = analyze_with_llama(PROMPT, text)
    try:
        # Use json.loads instead of eval for safer parsing
        return json.loads(result)
    except json.JSONDecodeError:
        # Fallback to regex pattern matching if direct parsing fails
        try:
            json_pattern = re.search(r'\{.*\}', result, re.DOTALL)
            if json_pattern:
                return json.loads(json_pattern.group(0))
        except:
            pass
        return {"error": "Analysis failed", "loaded_words": [], "missing_context": "", "slant_score": 5, "neutral_rewrites": []}

def enhance_context(text: str) -> str:
    """Add missing context intelligently."""
    PROMPT = """Improve this article by:
    1. Adjusting historical amounts for inflation
    2. Adding statistical baselines
    3. Flagging logical fallacies
    4. Inserting [CONTEXT] notes where needed
    
    Preserve the original text but make it more informative."""
    return analyze_with_llama(PROMPT, text)

def generate_perspectives(text: str) -> Dict[str, str]:
    """Generate balanced viewpoints with improved error handling"""
    PROMPT = """Provide 3 perspectives on this news article in JSON format.
    DO NOT include any text before or after the JSON object.
    ONLY return a valid JSON object with this exact structure:
    {
        "progressive": "Progressive viewpoint here with key arguments",
        "conservative": "Conservative counterarguments here",
        "expert": "Neutral expert analysis with citations when possible"
    }"""
    
    try:
        # Log the request initiation
        st.session_state['perspective_status'] = "Sending request to Llama 3.3..."
        
        # Get model response with reduced token count for stability
        result = analyze_with_llama(PROMPT, text[:3000])
        
        # Log response received
        st.session_state['perspective_status'] = "Processing response..."
        
        # Find JSON content and parse properly
        json_pattern = re.search(r'\{.*\}', result, re.DOTALL)
        if json_pattern:
            json_str = json_pattern.group(0)
            try:
                # Try direct parsing first
                return json.loads(json_str)
            except json.JSONDecodeError:
                # If direct parsing fails, try cleaning the JSON string
                cleaned_json = re.sub(r'\\n', ' ', json_str)
                cleaned_json = re.sub(r'\\', '', cleaned_json)
                try:
                    return json.loads(cleaned_json)
                except:
                    st.session_state['perspective_status'] = "Failed to parse JSON response"
        
        # If pattern matching failed
        st.session_state['perspective_status'] = "Could not extract valid JSON from response"
        return {
            "progressive": "Could not generate progressive perspective due to format issues. Please try again.",
            "conservative": "Could not generate conservative perspective due to format issues. Please try again.",
            "expert": "Could not generate expert analysis due to format issues. Please try again."
        }
            
    except Exception as e:
        error_msg = f"Perspective generation failed: {str(e)}"
        st.session_state['perspective_status'] = error_msg
        return {
            "progressive": "Could not generate progressive perspective. Technical error occurred.",
            "conservative": "Could not generate conservative perspective. Technical error occurred.",
            "expert": "Could not generate expert analysis. Technical error occurred."
        }

# --- STREAMLIT UI --- #
def main():
    st.title("📰 Objective News Translator")
    st.caption(f"Powered by Llama 3.3 70B via Groq • API Key: {'✅ Loaded' if GROQ_API_KEY else '❌ Missing'}")
    
    # Initialize session state for status tracking
    if 'perspective_status' not in st.session_state:
        st.session_state['perspective_status'] = ""
    
    # Sidebar controls
    with st.sidebar:
        st.header("Settings")
        analysis_mode = st.radio(
            "Analysis Depth",
            ["Fast (Basic NLP)", "Standard (Llama 3.3)", "Comprehensive (Full Fact-Check)"],
            index=1
        )
        st.info("💡 For best results, use Llama 3.3 mode with Groq's 300+ tokens/sec speed.")
        
        # Debug info
        st.divider()
        st.write("Debug Info:")
        st.code(f"GROQ_API_KEY: {'*****' + GROQ_API_KEY[-4:] if GROQ_API_KEY else 'Not found'}")
        
        # Status indicator
        if st.session_state['perspective_status']:
            st.write("Processing Status:")
            st.info(st.session_state['perspective_status'])
    
    # Article input
    url = st.text_input("Enter news article URL (NYT, WSJ, etc.)", "")
    
    if url:
        with st.spinner("🔄 Fetching article..."):
            try:
                article = Article(url)
                article.download()
                article.parse()
                
                # Store in session state
                st.session_state.original_text = article.text
                st.session_state.title = article.title
                st.session_state.authors = article.authors
                st.session_state.publish_date = article.publish_date
                
            except Exception as e:
                st.error(f"Failed to fetch article: {str(e)}")
                st.stop()
            
        # Show metadata
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(st.session_state.title)
            if st.session_state.authors:
                st.caption(f"By {', '.join(st.session_state.authors)}")
            if st.session_state.publish_date:
                st.caption(f"Published: {st.session_state.publish_date.strftime('%B %d, %Y')}")
        with col2:
            st.metric("Words", len(st.session_state.original_text.split()))
        
        # Analysis tabs
        tab1, tab2, tab3 = st.tabs(["Bias Analysis", "Enhanced Version", "Multi-Perspective"])
        
        with tab1:
            st.header("🔍 Bias Detection")
            with st.expander("Original Text", expanded=False):
                st.write(st.session_state.original_text)
            
            if analysis_mode != "Fast (Basic NLP)":
                with st.spinner("🧠 Analyzing with Llama 3.3 70B..."):
                    bias_report = detect_biases(st.session_state.original_text)
                    
                if "error" not in bias_report:
                    # Visualize loaded language
                    st.subheader("⚠️ Loaded Language")
                    loaded_words = bias_report.get("loaded_words", [])
                    if loaded_words:
                        st.write("These words may carry emotional/subjective weight:")
                        st.write(", ".join(f"`{w}`" for w in loaded_words))
                    else:
                        st.success("No strongly biased language detected")
                    
                    # Slant analysis
                    st.subheader("📊 Political Slant")
                    slant = bias_report.get("slant_score", 5)
                    fig = px.bar(x=["Left", "Center", "Right"], y=[max(0, 5-slant), 10-abs(5-slant)*2, max(0, slant-5)],
                                labels={"x": "", "y": "Intensity"}, color_discrete_sequence=["blue", "gray", "red"])
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Neutral rewrites
                    st.subheader("✏️ Suggested Neutral Phrases")
                    for rewrite in bias_report.get("neutral_rewrites", []):
                        if isinstance(rewrite, dict):
                            st.write(f"- **Original:** {rewrite.get('original', '')}")
                            st.write(f"- **Neutral:** {rewrite.get('rewrite', '')}")
                        else:
                            st.write(f"- {rewrite}")
                        st.divider()
            else:
                st.warning("Basic NLP mode selected. Upgrade to Llama 3.3 for deeper analysis.")
        
        with tab2:
            st.header("✨ Enhanced Version")
            if analysis_mode != "Fast (Basic NLP)":
                with st.spinner("📚 Adding missing context..."):
                    enhanced = enhance_context(st.session_state.original_text)
                    st.session_state.enhanced_text = enhanced
                
                # Side-by-side comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original")
                    st.write(st.session_state.original_text)
                with col2:
                    st.subheader("Enhanced")
                    st.write(st.session_state.enhanced_text)
                
                # Download option
                st.download_button(
                    label="📥 Download Enhanced Version",
                    data=st.session_state.enhanced_text,
                    file_name="enhanced_article.txt"
                )
            else:
                st.warning("Basic mode only shows original text. Upgrade for context enhancement.")
        
        with tab3:
            st.header("🔄 Multi-Perspective Analysis")
            if analysis_mode != "Fast (Basic NLP)":
                # Reset status
                st.session_state['perspective_status'] = "Preparing to generate perspectives..."
                
                # Add debug expander
                with st.expander("Debug Information", expanded=False):
                    st.write("This section shows the process of generating perspectives.")
                    st.write(f"Current status: {st.session_state['perspective_status']}")
                    if st.button("Test LLM Connection"):
                        test_response = analyze_with_llama("Respond with 'Connection successful!'", "Test")
                        st.write(f"LLM Response: {test_response}")
                
                with st.spinner("🤝 Generating balanced perspectives..."):
                    try:
                        # Show live status updates
                        status_placeholder = st.empty()
                        status_placeholder.info("Sending request to Llama 3.3...")
                        
                        # Generate perspectives
                        perspectives = generate_perspectives(st.session_state.original_text)
                        
                        # Update status
                        status_placeholder.success("Perspectives generated!")
                    except Exception as e:
                        st.error(f"Exception during generation: {str(e)}")
                        perspectives = {"error": "Failed to generate perspectives"}
                
                # Display debug information
                st.write(f"Final status: {st.session_state['perspective_status']}")
                
                if "error" not in perspectives:
                    for perspective, text in perspectives.items():
                        with st.expander(f"🔷 {perspective.capitalize()} View", expanded=True):
                            st.write(text)
                            st.divider()
                else:
                    st.error("Failed to generate perspectives")
                    st.write("Try refreshing or using a different article.")
            else:
                st.warning("Basic mode doesn't support multi-perspective. Use Llama 3.3.")
    
    else:
        st.info("👉 Enter a news URL to begin analysis")

if __name__ == "__main__":
    main()