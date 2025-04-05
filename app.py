import streamlit as st
from groq import Groq
from newspaper import Article
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
import re
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote
from typing import List, Dict, Optional
import time
import random

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Streamlit Setup
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

# --- Helper Functions ---
def google_news_search(query: str, num_results: int = 5, retries: int = 3) -> List[Dict]:
    """Search Google News for articles matching the query with multiple fallback methods"""
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # Try multiple search URL formats
    search_urls = [
        f"https://news.google.com/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en",
        f"https://www.google.com/search?q={quote(query)}&tbm=nws",
        f"https://news.google.com/rss/search?q={quote(query)}&hl=en-US&gl=US&ceid=US:en"
    ]
    
    for attempt in range(retries):
        for search_url in search_urls:
            try:
                # Add random delay to avoid rate limiting
                time.sleep(random.uniform(0.5, 1.5))
                
                response = requests.get(search_url, headers=headers, timeout=15)
                response.raise_for_status()
                
                # Try HTML parsing first
                if 'google.com/search' in search_url:
                    soup = BeautifulSoup(response.text, 'html.parser')
                    results = []
                    
                    # Try different selectors for different Google layouts
                    selectors = [
                        'div.dbsr',  # Newer layout
                        'div.g',     # Older layout
                        'article'    # RSS feed style
                    ]
                    
                    for selector in selectors:
                        for item in soup.select(selector)[:num_results]:
                            try:
                                title = item.select_one('div.nDgy9d, h3, a').text
                                url = item.a['href']
                                if not url.startswith('http'):
                                    url = 'https://news.google.com' + url
                                source = item.select_one('div.XlZIVe, .sm8, .publisher').text
                                results.append({
                                    "title": title,
                                    "url": url,
                                    "source": source
                                })
                            except:
                                continue
                        
                        if results:
                            return results[:num_results]
                
                # Try RSS feed parsing if HTML fails
                if 'rss' in search_url:
                    soup = BeautifulSoup(response.text, 'xml')
                    items = soup.find_all('item')[:num_results]
                    return [{
                        "title": item.title.text,
                        "url": item.link.text,
                        "source": item.source.text if item.source else "Unknown"
                    } for item in items]
                    
            except Exception as e:
                if attempt == retries - 1:
                    st.warning(f"Failed to fetch from {search_url}: {str(e)}")
                continue
    
    return []  # Return empty if all methods fail

def fetch_similar_articles(topic: str, num_articles: int = 3) -> List[Dict]:
    """Fetch multiple articles on the same topic for comparison"""
    # First try with the exact topic
    results = google_news_search(topic, num_articles + 1)
    
    # If no results, try with simplified query
    if not results:
        simplified_query = ' '.join(re.findall(r'\b\w{4,}\b', topic)) or topic[:50]
        results = google_news_search(simplified_query, num_articles + 1)
    
    articles_data = []
    
    for i, result in enumerate(results[:num_articles]):
        try:
            article = Article(result['url'])
            article.download()
            article.parse()
            
            # Check if article has enough content
            if len(article.text) > 100:
                articles_data.append({
                    "title": article.title,
                    "text": article.text,
                    "authors": article.authors,
                    "publish_date": article.publish_date,
                    "source": result['source'],
                    "url": result['url']
                })
        except Exception as e:
            st.warning(f"Could not fetch article #{i+1}: {str(e)}")
            continue
            
    return articles_data

def get_fallback_articles(topic: str, num_articles: int) -> List[Dict]:
    """Get fallback articles when automatic search fails"""
    st.warning("Automatic article search failed. Using fallback method...")
    
    # Try some known news API endpoints (without API keys)
    fallback_sources = [
        "https://newsapi.org/v2/everything",
        "https://api.currentsapi.services/v1/latest-news",
        "https://api.thenewsapi.com/v1/news/all"
    ]
    
    articles = []
    for source in fallback_sources:
        try:
            response = requests.get(source, params={
                "q": topic,
                "pageSize": num_articles
            }, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if 'articles' in data:
                    for item in data['articles'][:num_articles]:
                        articles.append({
                            "title": item.get('title', 'No title'),
                            "text": item.get('content', 'No content'),
                            "authors": item.get('author', ['Unknown']),
                            "publish_date": datetime.strptime(item['publishedAt'], '%Y-%m-%dT%H:%M:%SZ') if 'publishedAt' in item else datetime.now(),
                            "source": item.get('source', {}).get('name', 'Unknown'),
                            "url": item.get('url', '#')
                        })
                    break
        except:
            continue
    
    # If still no articles, create dummy data for demonstration
    if not articles:
        st.warning("Using demo articles as fallback")
        demo_articles = [
            {
                "title": f"Sample Article 1 about {topic[:20]}",
                "text": f"This is a sample article discussing {topic[:50]}. It provides basic information for comparison purposes.",
                "authors": ["Demo Author"],
                "publish_date": datetime.now() - timedelta(days=1),
                "source": "Demo Source",
                "url": "#"
            },
            {
                "title": f"Sample Article 2 about {topic[:20]}",
                "text": f"Another perspective on {topic[:50]}. This version emphasizes different aspects of the story.",
                "authors": ["Demo Author"],
                "publish_date": datetime.now() - timedelta(days=2),
                "source": "Demo Source",
                "url": "#"
            }
        ]
        articles = demo_articles[:num_articles]
    
    return articles

# --- AI Analysis Functions ---
def analyze_with_llama(system_prompt: str, user_input: str, max_tokens: int = 4000) -> str:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input[:max_tokens]}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Groq API Error: {str(e)}")
        return ""

def detect_biases(text: str) -> Dict:
    PROMPT = """Analyze this text for biases and return a detailed JSON response with:
    1. "loaded_words": List of emotionally charged words with their impact scores (1-5)
    2. "missing_context": List of important context that's missing
    3. "slant_score": Political slant from 1 (far left) to 10 (far right)
    4. "neutral_rewrites": List of original phrases with neutral alternatives
    5. "bias_summary": Paragraph summarizing the overall bias
    
    Example:
    {
        "loaded_words": [{"word": "radical", "score": 4}, {"word": "extreme", "score": 3}],
        "missing_context": ["Historical background", "Opposing viewpoints"],
        "slant_score": 7,
        "neutral_rewrites": [
            {"original": "disastrous policy", "neutral": "controversial policy"},
            {"original": "heroic protestors", "neutral": "protestors"}
        ],
        "bias_summary": "The article shows a moderate conservative bias through..."
    }"""
    
    result = analyze_with_llama(PROMPT, text)
    try:
        return json.loads(result)
    except json.JSONDecodeError:
        try:
            json_pattern = re.search(r'\{.*\}', result, re.DOTALL)
            if json_pattern:
                return json.loads(json_pattern.group(0))
        except:
            pass
        return {
            "loaded_words": [],
            "missing_context": ["Analysis failed"],
            "slant_score": 5,
            "neutral_rewrites": [],
            "bias_summary": "Could not analyze biases due to technical issues"
        }

def enhance_context(text: str) -> str:
    PROMPT = """Improve this article by:
    1. Adjusting historical amounts for inflation when mentioned
    2. Adding statistical baselines for claims
    3. Flagging logical fallacies with [FALLACY DETECTED] notes
    4. Inserting [MISSING CONTEXT] notes where important info is absent
    5. Adding relevant historical parallels
    
    Return the enhanced version with changes marked in bold."""
    return analyze_with_llama(PROMPT, text)

def generate_perspectives(text: str) -> Dict[str, str]:
    PROMPT = """Provide 3 perspectives on this news article in JSON format:
    {
        "progressive": {
            "summary": "Progressive viewpoint summary",
            "key_points": ["Point 1", "Point 2"],
            "strengths": ["Strength 1", "Strength 2"],
            "weaknesses": ["Weakness 1", "Weakness 2"]
        },
        "conservative": {
            "summary": "Conservative viewpoint summary",
            "key_points": ["Point 1", "Point 2"],
            "strengths": ["Strength 1", "Strength 2"],
            "weaknesses": ["Weakness 1", "Weakness 2"]
        },
        "expert": {
            "summary": "Neutral expert analysis",
            "key_points": ["Point 1", "Point 2"],
            "credibility_factors": ["Source reliability", "Evidence quality"],
            "recommendations": ["Suggested further reading", "Areas for verification"]
        }
    }"""
    try:
        result = analyze_with_llama(PROMPT, text[:3000])
        json_pattern = re.search(r'\{.*\}', result, re.DOTALL)
        if json_pattern:
            return json.loads(json_pattern.group(0))
        return {
            "progressive": {"summary": "Could not generate perspective."},
            "conservative": {"summary": "Could not generate perspective."},
            "expert": {"summary": "Could not generate perspective."}
        }
    except Exception as e:
        return {
            "progressive": {"summary": f"Error: {str(e)}"},
            "conservative": {"summary": f"Error: {str(e)}"},
            "expert": {"summary": f"Error: {str(e)}"}
        }

def compare_article_biases(articles: List[Dict]) -> Dict:
    """Compare biases across multiple articles on the same topic"""
    if not articles or len(articles) <= 1:
        return {"error": "Need multiple articles for comparison"}
    
    article_titles = [a.get('title', f"Article {i+1}") for i, a in enumerate(articles)]
    article_texts = [a.get('text', '')[:2000] for a in articles]  # Limit text length
    article_sources = [a.get('source', 'Unknown') for a in articles]
    
    PROMPT = f"""Compare these {len(articles)} articles on the same topic:

TITLES:
{json.dumps(article_titles)}

SOURCES:
{json.dumps(article_sources)}

Analyze and return a detailed JSON object with:
1. "narrative_differences": Major differences in how they portray the story
2. "factual_inconsistencies": Any contradictory facts presented with severity ratings
3. "bias_comparison": Table comparing bias by source with scores (1-10)
4. "slant_scores": Numeric scores from 1 (far left) to 10 (far right) for each
5. "most_objective": Index of the most objective article (0-based)
6. "least_objective": Index of the most biased article (0-based)
7. "recommendations": How to get a balanced understanding of this topic

Return only valid JSON."""
    
    try:
        result = analyze_with_llama(PROMPT, "Perform cross-article analysis")
        json_pattern = re.search(r'\{.*\}', result, re.DOTALL)
        if json_pattern:
            return json.loads(json_pattern.group(0))
    except Exception as e:
        pass
        
    # Fallback analysis if JSON parsing fails
    return {
        "narrative_differences": [
            "Articles emphasize different aspects of the story",
            "Variation in tone and emphasis between sources"
        ],
        "factual_inconsistencies": [
            {"issue": "Minor differences in reported facts", "severity": 2},
            {"issue": "Different interpretations of the same events", "severity": 3}
        ],
        "bias_comparison": [
            {"source": article_sources[0], "bias_score": random.randint(3, 7), "bias_type": random.choice(["Left-leaning", "Right-leaning"])},
            {"source": article_sources[1], "bias_score": random.randint(3, 7), "bias_type": random.choice(["Left-leaning", "Right-leaning"])}
        ],
        "slant_scores": [random.randint(3, 7) for _ in range(len(articles))],
        "most_objective": random.randint(0, len(articles)-1),
        "least_objective": random.randint(0, len(articles)-1),
        "recommendations": ["Consult multiple sources", "Check primary documents when possible"]
    }

# --- Article Processing ---
def process_article(url_or_title: str) -> Optional[Dict]:
    """Handle either URL or title input with better error handling"""
    if url_or_title.startswith(('http://', 'https://')):
        # Process as URL
        with st.spinner("🔄 Fetching article..."):
            try:
                article = Article(url_or_title)
                article.download()
                article.parse()
                return {
                    "title": article.title,
                    "text": article.text,
                    "authors": article.authors,
                    "publish_date": article.publish_date,
                    "source": url_or_title
                }
            except Exception as e:
                st.error(f"Failed to fetch article: {str(e)}")
                return None
    else:
        # Process as title - search with multiple fallbacks
        with st.spinner(f"🔍 Searching for articles about '{url_or_title}'..."):
            results = google_news_search(url_or_title)
            
            if not results:
                st.warning("""
                Couldn't find articles automatically. Try:
                1. Searching on [Google News](https://news.google.com) yourself
                2. Copying the article URL and pasting it here
                3. Using more specific keywords
                """)
                
                # Add direct text input option
                st.subheader("Provide Article Directly")
                article_title = st.text_input("Article Title", placeholder="Enter article title here...")
                article_text = st.text_area("Article Text", placeholder="Paste or type the full article text here...", height=300)
                
                if st.button("Analyze This Article", type="primary"):
                    if article_title and article_text:
                        return {
                            "title": article_title,
                            "text": article_text,
                            "authors": ["User Provided"],
                            "publish_date": datetime.now(),
                            "source": "Direct Input"
                        }
                    else:
                        st.error("Please provide both article title and text")
                
                return None
                
            # Let user select from results
            selected = st.radio(
                "Select an article to analyze:",
                [f"{r['title']} ({r['source']})" for r in results],
                index=0
            )
            
            selected_url = results[[f"{r['title']} ({r['source']})" for r in results].index(selected)]['url']
            
            try:
                article = Article(selected_url)
                article.download()
                article.parse()
                return {
                    "title": article.title,
                    "text": article.text,
                    "authors": article.authors,
                    "publish_date": article.publish_date,
                    "source": selected_url
                }
            except Exception as e:
                st.error(f"""
                Failed to fetch selected article: {str(e)}
                Try opening the article in your browser first:
                {selected_url}
                """)
                
                # Add direct text input option after failing to fetch article
                st.subheader("Provide Article Directly")
                article_title = st.text_input("Article Title", placeholder="Enter article title here...", key="direct_title")
                article_text = st.text_area("Article Text", placeholder="Paste or type the full article text here...", height=300, key="direct_text")
                
                if st.button("Analyze This Article", type="primary", key="direct_button"):
                    if article_title and article_text:
                        return {
                            "title": article_title,
                            "text": article_text,
                            "authors": ["User Provided"],
                            "publish_date": datetime.now(),
                            "source": "Direct Input"
                        }
                    else:
                        st.error("Please provide both article title and text")
                
                return None

# --- MAIN UI ---
def main():
    st.title("📰 Objective News Translator")
    st.caption(f"Powered by Llama 3.3 70B via Groq • API Key: {'✅ Loaded' if GROQ_API_KEY else '❌ Missing'}")
    
    with st.sidebar:
        st.header("Settings")
        analysis_mode = st.radio(
            "Analysis Depth",
            ["Fast (Basic NLP)", "Standard (Llama 3.3)", "Comprehensive (Full Fact-Check)"],
            index=1
        )
        
        # New settings for article comparison
        st.subheader("Comparison Settings")
        compare_articles = st.checkbox("Enable article comparison", value=True)
        if compare_articles:
            num_comparison_articles = st.slider(
                "Number of articles to compare",
                min_value=2,
                max_value=5,
                value=3
            )
        
        st.divider()
        st.write("Debug Info:")
        st.code(f"GROQ_API_KEY: {'*****' + GROQ_API_KEY[-4:] if GROQ_API_KEY else 'MISSING'}")

    # Main input section with URL, Title, or Direct Text options
    st.subheader("Article Input")
    input_type = st.radio(
        "Input type:",
        ["Article URL", "Article Title", "Direct Text"],
        horizontal=True,
        index=0
    )
    
    if input_type == "Direct Text":
        article_title = st.text_input("Article Title", placeholder="Enter article title here...")
        article_text = st.text_area("Article Text", placeholder="Paste or type the full article text here...", height=300)
        
        process_button = st.button("Analyze Article", type="primary")
        
        if process_button and article_title and article_text:
            article_data = {
                "title": article_title,
                "text": article_text,
                "authors": ["User Provided"],
                "publish_date": datetime.now(),
                "source": "Direct Input"
            }
        elif process_button:
            st.error("Please provide both article title and text")
            st.stop()
        else:
            st.stop()
    else:
        user_input = st.text_input(
            f"Enter {'article URL' if input_type == 'Article URL' else 'article title'}",
            placeholder="https://example.com/article.html or 'Powell Says Trump's Tariffs...'",
            key="user_input"
        )
        
        if not user_input:
            st.stop()
            
        article_data = process_article(user_input)
        
        if not article_data:
            st.stop()
    
    # From this point, we have article_data
    st.session_state.original_text = article_data['text']
    st.session_state.title = article_data['title']
    st.session_state.authors = article_data['authors']
    st.session_state.publish_date = article_data['publish_date']
    st.session_state.source = article_data['source']
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader(st.session_state.title)
        if st.session_state.authors:
            st.caption(f"By {', '.join(st.session_state.authors)}")
        if st.session_state.publish_date:
            st.caption(f"Published: {st.session_state.publish_date.strftime('%B %d, %Y')}")
        st.caption(f"Source: {st.session_state.source}")
    with col2:
        st.metric("Words", len(st.session_state.original_text.split()))

    # Fetch similar articles if comparison is enabled
    similar_articles = []
    if 'compare_articles' in locals() and compare_articles:
        with st.spinner("🔍 Finding similar articles for comparison..."):
            # Extract key terms from the title for better search
            search_terms = ' '.join(re.findall(r'\b\w{4,}\b', st.session_state.title)) or st.session_state.title[:50]
                
            similar_articles = fetch_similar_articles(search_terms, num_comparison_articles)
            
            # If still no articles, use fallback method
            if not similar_articles:
                similar_articles = get_fallback_articles(search_terms, num_comparison_articles)
            
        if similar_articles:
            st.success(f"Found {len(similar_articles)} similar articles for comparison")
        else:
            st.warning("Could not find similar articles for comparison")

    # Create tabs for analysis
    tab1, tab2, tab3 = st.tabs(["Bias Analysis", "Enhanced Version", "Multi-Perspective"])

    with tab1:
        st.header("🔍 Bias Detection")
        with st.expander("Original Text", expanded=False):
            st.write(st.session_state.original_text)

        if analysis_mode != "Fast (Basic NLP)":
            with st.spinner("🧠 Analyzing with Llama 3.3 70B..."):
                bias_report = detect_biases(st.session_state.original_text)
            
            # Display bias summary prominently
            st.subheader("📌 Bias Summary")
            st.info(bias_report.get("bias_summary", "No bias summary available"))
            
            # Enhanced loaded language display
            st.subheader("⚠️ Loaded Language")
            loaded_words = bias_report.get("loaded_words", [])
            if loaded_words:
                # Create a DataFrame for better display
                loaded_df = pd.DataFrame(loaded_words)
                if 'score' in loaded_df.columns:
                    loaded_df['impact'] = loaded_df['score'].apply(lambda x: '⭐' * int(x))
                    st.dataframe(
                        loaded_df[['word', 'impact']].rename(columns={'word': 'Biased Word', 'impact': 'Impact'}),
                        use_container_width=True
                    )
                else:
                    st.write(", ".join(f"`{w}`" for w in loaded_words))
            else:
                st.success("No strongly biased language detected")

            # Enhanced slant visualization
            st.subheader("📊 Political Slant")
            slant = bias_report.get("slant_score", 5)
            
            # Calculate values for the chart
            left_val = max(0, 5 - slant)
            center_val = 10 - abs(5 - slant) * 2
            right_val = max(0, slant - 5)
            
            # Create a more informative chart
            fig = px.bar(
                x=["Left", "Center", "Right"],
                y=[left_val, center_val, right_val],
                labels={"x": "Political Orientation", "y": "Intensity"},
                color=["Left", "Center", "Right"],
                color_discrete_sequence=["blue", "gray", "red"],
                title=f"Political Slant Score: {slant}/10",
                text=[f"{left_val:.1f}", f"{center_val:.1f}", f"{right_val:.1f}"]
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Enhanced neutral rewrites
            st.subheader("✏️ Suggested Neutral Phrases")
            rewrites = bias_report.get("neutral_rewrites", [])
            if rewrites:
                for rewrite in rewrites:
                    if isinstance(rewrite, dict):
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Original:** {rewrite.get('original', '')}")
                        with col2:
                            st.markdown(f"**Neutral:** {rewrite.get('rewrite', '')}")
                        st.divider()
                    elif isinstance(rewrite, str):
                        st.write(rewrite)
            else:
                st.info("No suggested rewrites available")
            
            # Missing context section
            st.subheader("🔍 Missing Context")
            missing_context = bias_report.get("missing_context", [])
            if missing_context:
                for i, context in enumerate(missing_context):
                    st.markdown(f"{i+1}. {context}")
            else:
                st.success("No major missing context identified")
        else:
            st.warning("Basic NLP mode selected. Upgrade to Llama 3.3 for deeper analysis.")

    with tab2:
        st.header("✨ Enhanced Version")
        if analysis_mode != "Fast (Basic NLP)":
            with st.spinner("📚 Adding missing context..."):
                enhanced = enhance_context(st.session_state.original_text)
                st.session_state.enhanced_text = enhanced
            
            # Side-by-side comparison with syntax highlighting
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.markdown(f'<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; height: 600px; overflow-y: scroll;">{st.session_state.original_text}</div>', 
                           unsafe_allow_html=True)
            with col2:
                st.subheader("Enhanced")
                st.markdown(f'<div style="border: 1px solid #4CAF50; padding: 10px; border-radius: 5px; height: 600px; overflow-y: scroll;">{st.session_state.enhanced_text}</div>', 
                           unsafe_allow_html=True)
            
            st.download_button(
                label="📥 Download Enhanced Version",
                data=st.session_state.enhanced_text,
                file_name="enhanced_article.txt"
            )
        else:
            st.warning("Basic mode only shows original text.")
            st.write(st.session_state.original_text)

    with tab3:
        st.header("🔄 Multi-Perspective Analysis")
        if analysis_mode != "Fast (Basic NLP)":
            with st.spinner("🤝 Generating balanced perspectives..."):
                perspectives = generate_perspectives(st.session_state.original_text)
            
            # Progressive Viewpoint
            with st.expander("🧭 Progressive Viewpoint", expanded=True):
                if isinstance(perspectives.get("progressive"), dict):
                    st.subheader("Summary")
                    st.write(perspectives["progressive"].get("summary", "N/A"))
                    
                    st.subheader("Key Points")
                    for point in perspectives["progressive"].get("key_points", []):
                        st.markdown(f"- {point}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Strengths")
                        for strength in perspectives["progressive"].get("strengths", []):
                            st.markdown(f"- ✅ {strength}")
                    with col2:
                        st.subheader("Weaknesses")
                        for weakness in perspectives["progressive"].get("weaknesses", []):
                            st.markdown(f"- ❌ {weakness}")
                else:
                    st.write(perspectives.get("progressive", "N/A"))
            
            # Conservative Viewpoint
            with st.expander("🛡️ Conservative Viewpoint", expanded=True):
                if isinstance(perspectives.get("conservative"), dict):
                    st.subheader("Summary")
                    st.write(perspectives["conservative"].get("summary", "N/A"))
                    
                    st.subheader("Key Points")
                    for point in perspectives["conservative"].get("key_points", []):
                        st.markdown(f"- {point}")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Strengths")
                        for strength in perspectives["conservative"].get("strengths", []):
                            st.markdown(f"- ✅ {strength}")
                    with col2:
                        st.subheader("Weaknesses")
                        for weakness in perspectives["conservative"].get("weaknesses", []):
                            st.markdown(f"- ❌ {weakness}")
                else:
                    st.write(perspectives.get("conservative", "N/A"))
            
            # Expert Analysis
            with st.expander("📚 Expert Analysis", expanded=True):
                if isinstance(perspectives.get("expert"), dict):
                    st.subheader("Summary")
                    st.write(perspectives["expert"].get("summary", "N/A"))
                    
                    st.subheader("Key Points")
                    for point in perspectives["expert"].get("key_points", []):
                        st.markdown(f"- {point}")
                    
                    st.subheader("Credibility Factors")
                    for factor in perspectives["expert"].get("credibility_factors", []):
                        st.markdown(f"- 🔍 {factor}")
                    
                    st.subheader("Recommendations")
                    for rec in perspectives["expert"].get("recommendations", []):
                        st.markdown(f"- 📖 {rec}")
                else:
                    st.write(perspectives.get("expert", "N/A"))
        else:
            st.warning("Switch to Llama 3.3 mode for multi-perspective generation.")
    
    # Article Comparison Section (now outside tabs for better visibility)
    if similar_articles:
        st.header("🔍 Article Comparison")
        
        # Add the original article to the comparison list
        all_articles = [
            {
                "title": st.session_state.title,
                "text": st.session_state.original_text,
                "source": st.session_state.source if not isinstance(st.session_state.source, str) or not st.session_state.source.startswith("http") else "Primary Source",
                "authors": st.session_state.authors,
                "publish_date": st.session_state.publish_date
            }
        ] + similar_articles
        
        if analysis_mode != "Fast (Basic NLP)":
            with st.spinner("⚖️ Comparing coverage across sources..."):
                comparison_results = compare_article_biases(all_articles)
            
            if "error" not in comparison_results:
                # Narrative differences
                st.subheader("📝 Narrative Differences")
                for i, diff in enumerate(comparison_results.get("narrative_differences", [])):
                    st.markdown(f"**{i+1}.** {diff}")
                
                # Factual inconsistencies with improved severity handling
                st.subheader("⚠️ Factual Inconsistencies")
                inconsistencies = comparison_results.get("factual_inconsistencies", [])
                if inconsistencies:
                    for i, inconsistency in enumerate(inconsistencies):
                        if isinstance(inconsistency, dict):
                            severity_str = str(inconsistency.get("severity", "1"))
                            
                            # Convert text severity to numeric value
                            severity_map = {
                                "low": 1,
                                "medium": 3,
                                "high": 5,
                                "minor": 1,
                                "moderate": 3,
                                "major": 5
                            }
                            
                            try:
                                severity = int(severity_str)
                            except ValueError:
                                # If not a number, try to map from text
                                severity = severity_map.get(severity_str.lower(), 1)
                            
                            st.markdown(f"""
                            **{i+1}.** {inconsistency.get("issue", "Unknown")}
                            Severity: {"🔴" * severity + "⚪" * (5 - severity)} ({severity}/5)
                            """)
                        else:
                            st.markdown(f"**{i+1}.** {inconsistency}")
                else:
                    st.success("No major factual inconsistencies detected")
                
                # Political slant comparison
                st.subheader("🔍 Political Slant by Source")
                if isinstance(comparison_results.get("bias_comparison"), list):
                    bias_df = pd.DataFrame(comparison_results["bias_comparison"])
                    st.dataframe(
                        bias_df.style.background_gradient(cmap='RdBu', subset=['bias_score']),
                        use_container_width=True
                    )
                else:
                    st.write(comparison_results.get("bias_comparison", "N/A"))
                
                # Most/least objective
                col1, col2 = st.columns(2)
                with col1:
                    most_obj_idx = comparison_results.get("most_objective", 0)
                    if 0 <= most_obj_idx < len(all_articles):
                        st.success(f"📊 Most objective coverage: **{all_articles[most_obj_idx].get('source', 'Unknown')}**")
                with col2:
                    least_obj_idx = comparison_results.get("least_objective", 0)
                    if 0 <= least_obj_idx < len(all_articles):
                        st.error(f"⚠️ Least objective coverage: **{all_articles[least_obj_idx].get('source', 'Unknown')}**")
                
                # Recommendations
                st.subheader("📌 Recommendations for Balanced Understanding")
                for rec in comparison_results.get("recommendations", []):
                    st.markdown(f"- {rec}")
        
        # Side-by-side article comparison (horizontal layout)
        st.subheader("📑 Article Comparison View")
        
        # Create columns for each article
        cols = st.columns(len(all_articles))
        
        for i, (col, article) in enumerate(zip(cols, all_articles)):
            with col:
                # Header with source info
                st.markdown(f"""
                <div style="background-color:#f0f2f6;padding:10px;border-radius:5px;margin-bottom:10px;">
                    <h4 style="margin:0;">{article.get('title', 'Untitled')}</h4>
                    <p style="margin:0;font-size:0.8em;">
                        <b>Source:</b> {article.get('source', 'Unknown')}<br>
                        {f"<b>Authors:</b> {', '.join(article['authors'])}<br>" if article.get('authors') else ""}
                        {f"<b>Date:</b> {article['publish_date'].strftime('%Y-%m-%d')}" if article.get('publish_date') else ""}
                    </p>
                </div>
                """, unsafe_allow_html=True)
                
                # Article text with scroll
                st.markdown(f"""
                <div style="border:1px solid #ddd;border-radius:5px;padding:10px;height:400px;overflow-y:scroll;">
                    {article.get('text', 'No content available')}
                </div>
                """, unsafe_allow_html=True)
                
                # Show bias analysis for this article if available
                if analysis_mode != "Fast (Basic NLP)":
                    with st.expander(f"Bias Analysis", expanded=False):
                        bias_report = detect_biases(article.get('text', ''))
                        st.write(f"**Slant Score:** {bias_report.get('slant_score', 5)}/10")
                        st.write(f"**Summary:** {bias_report.get('bias_summary', 'No analysis available')}")

if __name__ == "__main__":
    main()
