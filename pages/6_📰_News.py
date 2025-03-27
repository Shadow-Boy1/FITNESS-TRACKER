import streamlit as st
import requests

# News API key (Replace 'YOUR_NEWSAPI_KEY' with your actual API key)
API_KEY = "d01adc1d8ec6466f94be3d2a7fb66a94"
BASE_URL = "https://newsapi.org/v2/everything"

def fetch_fitness_news(query="fitness"):
    """Fetch fitness news based on the query"""
    try:
        params = {
            "q": query,  
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": API_KEY,
        }
        response = requests.get(BASE_URL, params=params)
        data = response.json()
        if data.get("status") == "ok":
            return data.get("articles", [])[:10]  # Get top 10 articles
        else:
            st.error("Failed to fetch news. Please try again.")
            return []
    except requests.exceptions.RequestException:
        st.error("Network error. Check your internet connection.")
        return []

# 🌟 Streamlit UI
st.set_page_config(page_title="Fitness News", page_icon="🏋️‍♂️")
st.title("🏋️‍♂️ Fitness News")

# 🔍 Search bar
search_query = st.text_input("Search Fitness News", "fitness")
if st.button("Search"):
    news_articles = fetch_fitness_news(search_query)
else:
    news_articles = fetch_fitness_news()

# 📰 Display news articles
if news_articles:
    for article in news_articles:
        st.subheader(article["title"])
        st.write(article.get("description", "No description available."))
        st.markdown(f"[Read more]({article['url']})", unsafe_allow_html=True)
        if article.get("urlToImage"):
            st.image(article["urlToImage"], use_container_width=True)
        st.write("---")
else:
    st.warning("No fitness news available at the moment.")
