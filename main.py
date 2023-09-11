import requests
from bs4 import BeautifulSoup
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import streamlit as st


class WebScraper:
    def __init__(self):
        self.url = None  # Store the URL to scrape

    def scrape_website(self, url):
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text_data = soup.select_one('div.article-content').get_text()
        return text_data

    def clean_data(self, data):
        cleaned_data = re.sub(r'\d+', '', data)
        return cleaned_data

    def handle_duplicates(self, data):
        unique_data = list(set(data))
        return unique_data

    def handle_missing_values(self, data):
        filled_data = list(filter(None, data))
        return filled_data

    def normalize_data(self, data):
        blob = TextBlob(data)
        normalized_data = ' '.join([word.lemmatize() for word in blob.words])
        return normalized_data

    def perform_sentiment_analysis(self, data):
        blob = TextBlob(data)
        polarity = blob.sentiment.polarity
        if polarity > 0:
            return 'Positive'
        elif polarity < 0:
            return 'Negative'
        else:
            return 'Neutral'

    def extract_topics(self, data):
        vectorizer = CountVectorizer(
            max_df=0.95, min_df=2, stop_words='english')
        dtm = vectorizer.fit_transform(data)

        lda_model = LatentDirichletAllocation(n_components=5, random_state=42)
        lda_model.fit(dtm)

        topics = []
        for topic_idx, topic in enumerate(lda_model.components_):
            top_words = [vectorizer.get_feature_names()[i]
                         for i in topic.argsort()[:-6:-1]]
            topics.append(f"Topic {topic_idx + 1}: {' '.join(top_words)}")
        return topics

    def visualize_data(self, topics):
        x = range(len(topics))
        y = np.random.randint(0, 100, len(topics))
        plt.bar(x, y)
        plt.xlabel('Topics')
        plt.ylabel('Number of Documents')
        plt.title('Topics Distribution')
        plt.xticks(x, topics, rotation=45)
        plt.show()

    def run_web_scraper(self):
        self.url = st.text_input('Enter the URL to scrape:')

        if st.button('Scrape and Analyze'):
            with st.spinner('Scraping and analyzing data...'):
                scraped_data = self.scrape_website(self.url)
                cleaned_data = self.clean_data(scraped_data)
                unique_data = self.handle_duplicates(cleaned_data)
                filled_data = self.handle_missing_values(unique_data)
                normalized_data = self.normalize_data(' '.join(filled_data))
                sentiment = self.perform_sentiment_analysis(normalized_data)
                topics = self.extract_topics(normalized_data)
                self.visualize_data(topics)

        st.info(
            'Use this program to scrape and analyze data from various online sources.')


def main():
    st.title('Web Scraping and Analysis Program')

    scraper = WebScraper()
    scraper.run_web_scraper()


if __name__ == '__main__':
    main()
