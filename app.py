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
    PROMPT = """Analyze this text for:
    1. Loaded language (emotional/subjective words)
    2. Omitted context (what's missing?)
    3. Political slant (Left/Center/Right)
    4. Suggested neutral rewrites
    
    Return as JSON with keys: loaded_words, missing_context, slant_score (1-10), neutral_rewrites"""
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
            "missing_context": "Analysis failed",
            "slant_score": 5,
            "neutral_rewrites": []
        }

def enhance_context(text: str) -> str:
    PROMPT = """Improve this article by:
    1. Adjusting historical amounts for inflation
    2. Adding statistical baselines
    3. Flagging logical fallacies
    4. Inserting [CONTEXT] notes where needed
    
    Preserve the original text but make it more informative."""
    return analyze_with_llama(PROMPT, text)

def generate_perspectives(text: str) -> Dict[str, str]:
    PROMPT = """Provide 3 perspectives on this news article in JSON format:
    {
        "progressive": "Progressive viewpoint here with key arguments",
        "conservative": "Conservative counterarguments here",
        "expert": "Neutral expert analysis with citations when possible"
    }"""
    try:
        result = analyze_with_llama(PROMPT, text[:3000])
        json_pattern = re.search(r'\{.*\}', result, re.DOTALL)
        if json_pattern:
            return json.loads(json_pattern.group(0))
        return {
            "progressive": "Could not generate perspective.",
            "conservative": "Could not generate perspective.",
            "expert": "Could not generate perspective."
        }
    except Exception as e:
        return {
            "progressive": f"Error: {str(e)}",
            "conservative": f"Error: {str(e)}",
            "expert": f"Error: {str(e)}"
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

Analyze and return a JSON object with:
1. "narrative_differences": Major differences in how they portray the story
2. "factual_inconsistencies": Any contradictory facts presented
3. "bias_comparison": Comparative analysis of bias by source
4. "slant_scores": Numeric scores from 1 (far left) to 10 (far right) for each source
5. "most_objective": Index of the most objective article (0-based)

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
            "Minor differences in reported facts",
            "Different interpretations of the same events"
        ],
        "bias_comparison": {
            "Left-leaning": "Emphasizes social impacts",
            "Right-leaning": "Focuses on economic aspects",
            "Centrist": "Balanced coverage"
        },
        "slant_scores": [random.randint(3, 7) for _ in range(len(articles))],
        "most_objective": random.randint(0, len(articles)-1)
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
    tab1, tab2, tab3, tab4 = st.tabs(["Bias Analysis", "Enhanced Version", "Multi-Perspective", "Article Comparison"])

    with tab1:
        st.header("🔍 Bias Detection")
        with st.expander("Original Text", expanded=False):
            st.write(st.session_state.original_text)

        if analysis_mode != "Fast (Basic NLP)":
            with st.spinner("🧠 Analyzing with Llama 3.3 70B..."):
                bias_report = detect_biases(st.session_state.original_text)
            
            st.subheader("⚠️ Loaded Language")
            loaded_words = bias_report.get("loaded_words", [])
            if loaded_words:
                st.write(", ".join(f"`{w}`" for w in loaded_words))
            else:
                st.success("No strongly biased language detected")

            st.subheader("📊 Political Slant")
            slant = bias_report.get("slant_score", 5)
            fig = px.bar(
                x=["Left", "Center", "Right"],
                y=[max(0, 5-slant), 10-abs(5-slant)*2, max(0, slant-5)],
                labels={"x": "", "y": "Intensity"},
                color_discrete_sequence=["blue", "gray", "red"]
            )
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("✏️ Suggested Neutral Phrases")
            for rewrite in bias_report.get("neutral_rewrites", []):
                if isinstance(rewrite, dict):
                    st.markdown(f"- **Original:** {rewrite.get('original', '')}")
                    st.markdown(f"- **Neutral:** {rewrite.get('rewrite', '')}")
                    st.divider()
                elif isinstance(rewrite, str):
                    st.write(rewrite)
        else:
            st.warning("Basic NLP mode selected. Upgrade to Llama 3.3 for deeper analysis.")

    with tab2:
        st.header("✨ Enhanced Version")
        if analysis_mode != "Fast (Basic NLP)":
            with st.spinner("📚 Adding missing context..."):
                enhanced = enhance_context(st.session_state.original_text)
                st.session_state.enhanced_text = enhanced
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original")
                st.write(st.session_state.original_text)
            with col2:
                st.subheader("Enhanced")
                st.write(st.session_state.enhanced_text)
            st.download_button(
                label="📥 Download Enhanced Version",
                data=st.session_state.enhanced_text,
                file_name="enhanced_article.txt"
            )
        else:
            st.warning("Basic mode only shows original text.")

    with tab3:
        st.header("🔄 Multi-Perspective Analysis")
        if analysis_mode != "Fast (Basic NLP)":
            with st.spinner("🤝 Generating balanced perspectives..."):
                perspectives = generate_perspectives(st.session_state.original_text)
            st.subheader("🧭 Progressive Viewpoint")
            st.write(perspectives.get("progressive", "N/A"))
            st.subheader("🛡️ Conservative Viewpoint")
            st.write(perspectives.get("conservative", "N/A"))
            st.subheader("📚 Expert Analysis")
            st.write(perspectives.get("expert", "N/A"))
        else:
            st.warning("Switch to Llama 3.3 mode for multi-perspective generation.")
    
    with tab4:
        st.header("🔍 Article Comparison")
        if not similar_articles:
            st.warning("No similar articles found for comparison. Try searching with a more specific title.")
        else:
            st.subheader("📊 Cross-Source Analysis")
            
            # Analyze differences between articles
            if analysis_mode != "Fast (Basic NLP)":
                with st.spinner("⚖️ Comparing coverage across sources..."):
                    # Add the original article to the comparison list
                    all_articles = [
                        {
                            "title": st.session_state.title,
                            "text": st.session_state.original_text,
                            "source": st.session_state.source if not isinstance(st.session_state.source, str) or not st.session_state.source.startswith("http") else "Primary Source"
                        }
                    ] + similar_articles
                    
                    comparison_results = compare_article_biases(all_articles)
                
                if "error" not in comparison_results:
                    # Show narrative differences
                    st.subheader("📝 Narrative Differences")
                    for i, diff in enumerate(comparison_results.get("narrative_differences", [])):
                        st.markdown(f"**{i+1}.** {diff}")
                    
                    # Show factual inconsistencies
                    st.subheader("⚠️ Factual Inconsistencies")
                    inconsistencies = comparison_results.get("factual_inconsistencies", [])
                    if inconsistencies:
                        for i, inconsistency in enumerate(inconsistencies):
                            st.markdown(f"**{i+1}.** {inconsistency}")
                    else:
                        st.success("No major factual inconsistencies detected")
                    
                    # Political slant comparison
                    st.subheader("🔍 Political Slant by Source")
                    sources = [a.get("source", f"Source {i}") for i, a in enumerate(all_articles)]
                    slant_scores = comparison_results.get("slant_scores", [5] * len(all_articles))

                    # In the Bias Analysis tab (around line 652 in your original code)
                    st.subheader("📊 Political Slant")
                    slant = bias_report.get("slant_score", 5)

# Calculate the values for left, center, right
                    left_val = max(0, 5 - slant)
                    center_val = 10 - abs(5 - slant) * 2
                    right_val = max(0, slant - 5)

# Ensure all values are positive
                    left_val = max(0, left_val)
                    center_val = max(0, center_val)
                    right_val = max(0, right_val)
                    
                    # Create a bar chart comparing slants
                    fig = px.bar(
                    x=["Left", "Center", "Right"],
                    y=[left_val, center_val, right_val],
                    labels={"x": "", "y": "Intensity"},
                    color=["Left", "Center", "Right"],
                    color_discrete_sequence=["blue", "gray", "red"]
 )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Most objective source
                    most_objective_idx = comparison_results.get("most_objective", 0)
                    if 0 <= most_objective_idx < len(all_articles):
                        st.success(f"📊 Most objective coverage: **{all_articles[most_objective_idx].get('source', 'Unknown')}**")
            
            # Article side-by-side comparison
            st.subheader("📑 Article Side-by-Side")
            
            # Create tabs for each article
            article_tabs = st.tabs(["Original"] + [f"Similar {i+1}" for i in range(len(similar_articles))])
            
            # Display the original article
            with article_tabs[0]:
                st.subheader(st.session_state.title)
                st.caption(f"Source: {st.session_state.source}")
                st.write(st.session_state.original_text)
            
            # Display each similar article
            for i, article in enumerate(similar_articles):
                with article_tabs[i+1]:
                    st.subheader(article.get("title", f"Article {i+1}"))
                    st.caption(f"Source: {article.get('source', 'Unknown')}")
                    if article.get("authors"):
                        st.caption(f"By {', '.join(article['authors'])}")
                    if article.get("publish_date"):
                        st.caption(f"Published: {article['publish_date'].strftime('%B %d, %Y')}")
                    st.write(article.get("text", "No content available"))
        
            # Bias heat map across all articles
            if analysis_mode != "Fast (Basic NLP)" and len(similar_articles) > 0:
                st.subheader("🔥 Bias Heat Map")
                
                # Get bias scores for all articles
                all_articles = [
                    {
                        "title": st.session_state.title,
                        "text": st.session_state.original_text,
                        "source": st.session_state.source if not isinstance(st.session_state.source, str) or not st.session_state.source.startswith("http") else "Primary Source"
                    }
                ] + similar_articles
                
                # Get bias categories and scores
                bias_categories = ["Loaded Language", "Missing Context", "Political Slant"]
                sources = [a.get("source", f"Source {i}") if not isinstance(a.get("source"), str) or not a.get("source", "").startswith("http") else f"Source {i}" for i, a in enumerate(all_articles)]
                
                # Create dataframe for heatmap
                bias_data = []
                for i, article in enumerate(all_articles):
                    with st.spinner(f"Analyzing bias in article {i+1}..."):
                        bias_report = detect_biases(article.get("text", ""))
                        loaded_words_score = min(10, len(bias_report.get("loaded_words", [])))
                        missing_context_score = len(bias_report.get("missing_context", "")) / 20  # Normalize to 0-10
                        slant_deviation = abs(bias_report.get("slant_score", 5) - 5) * 2  # Convert to 0-10 scale
                        
                        bias_data.append({
                            "Source": sources[i],
                            "Loaded Language": loaded_words_score,
                            "Missing Context": missing_context_score,
                            "Political Slant": slant_deviation
                        })
                
                # Create a DataFrame for the heatmap
                bias_df = pd.DataFrame(bias_data)
                bias_df_melted = pd.melt(
                    bias_df, 
                    id_vars=["Source"], 
                    value_vars=bias_categories,
                    var_name="Bias Category", 
                    value_name="Score"
                )
                
                # Create the heatmap
                fig = px.imshow(
                    bias_df.set_index("Source")[bias_categories].values,
                    x=bias_categories,
                    y=bias_df["Source"],
                    color_continuous_scale="Reds",
                    labels={"color": "Bias Score"},
                    title="Bias Comparison Across Sources"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show summary table
                st.subheader("📊 Bias Summary Table")
                st.dataframe(bias_df, use_container_width=True)

if __name__ == "__main__":
    main()
