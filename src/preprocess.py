import os
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

import re

def preprocess_text(text):
    if isinstance(text, str):  # Check if the input is a string
        text = re.sub(r'\W', ' ', text.lower())
        return text
    return ''  # Return empty string for non-string inputs


def load_articles():
    articles = []
    try:

        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Construct the path to the data folder
        data_folder = os.path.join(current_dir, '..', 'data')

        df1 = pd.read_csv(os.path.join(data_folder, 'news_articles_1.csv'))

        # df1 = pd.read_csv('/data/news_articles_1.csv')
        for index, row in df1.iterrows():
            articles.append({
                'title': row['title'],
                'url': row['url'],
                'published_at': row['published_at'],
                'author': row['author'],
                'publisher': row['publisher'],
                'short_description': row['short_description'],
                'keywords': row['keywords'],
                'header_image': row['header_image'],
                'raw_description': row['raw_description'],
                'description': row['description'],
                'scraped_at': row['scraped_at'],
                'processed_content': preprocess_text(row['raw_description'])  # or another relevant column
            })

        # Load the second CSV file
        df2 = pd.read_csv(os.path.join(data_folder, 'news_articles_2.csv'), encoding='ISO-8859-1')

        # df2 = pd.read_csv('data/news_articles_2.csv')
        for index, row in df2.iterrows():
            articles.append({
                'title': row['Heading'],  # Assuming Heading is a title equivalent
                'published_at': row['Date'],
                'author': '',  # If no author info, you can leave it empty
                'url': '',  # If no URL info, you can leave it empty
                'description': row['Article'],  # or any other relevant column
                'processed_content': preprocess_text(row['Article'])
            })

        # Load the third CSV file
        df3 = pd.read_csv(os.path.join(data_folder, 'news_articles_3.csv'))

        # df3 = pd.read_csv('data/news_articles_3.csv')
        for index, row in df3.iterrows():
            articles.append({
                'title': row['Headline'],
                'published_at': row['Date published'],
                'author': row['Author'],
                'url': row['Url'],
                'description': row['Description'],
                'processed_content': preprocess_text(row['Article text'])
            })

        # Load the fourth CSV file
        df4 = pd.read_csv(os.path.join(data_folder, 'news_articles_4.csv'), encoding='ISO-8859-1')

        # df4 = pd.read_csv('data/news_articles_4.csv')
        for index, row in df4.iterrows():
            articles.append({
                'title': row['STORY'],  # Assuming STORY is a title equivalent
                'published_at': '',
                'author': '',
                'url': '',
                'description': '',
                'processed_content': preprocess_text(row['SECTION'])  # or another relevant column
            })

        return articles
    except FileNotFoundError as e:
        print(f"File not found: {e}")


# Example usage
if __name__ == "__main__":
    articles = load_articles()
    print(f"Loaded {len(articles)} articles.")
