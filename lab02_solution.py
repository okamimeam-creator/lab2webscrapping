"""
Lab 02 – Web Scraping, Feature Engineering, and Exploratory Data Analysis
Run from the project folder:
    python lab02_solution.py

The script scrapes the five URLs from the lab handout, extracts NLP features,
creates EDA plots, and saves CSV outputs.
"""

from pathlib import Path
import time
import warnings

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import spacy
import matplotlib
matplotlib.use("Agg")  # Save plots without requiring an interactive window
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

URLS = [
    "https://techncruncher.blogspot.com/2025/01/top-10-ai-tools-that-will-transform.html",
    "https://techncruncher.blogspot.com/2023/12/limewire-ai-studio-review-2023-details.html",
    "https://techncruncher.blogspot.com/2023/01/top-10-ai-tools-in-2023-that-will-make.html",
    "https://techncruncher.blogspot.com/2022/11/top-10-ai-content-generator-writer.html",
    "https://techncruncher.blogspot.com/2022/09/cj-affiliate-ultimate-guide-to.html",
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
}


def benchmark_numpy_vs_python():
    """Compare Python sum and NumPy sum on 10 million random numbers."""
    print("\n[1/6] Running NumPy vs Python summation benchmark...")
    values = np.random.default_rng(42).random(10_000_000)

    start = time.perf_counter()
    python_total = sum(values)
    python_seconds = time.perf_counter() - start

    start = time.perf_counter()
    numpy_total = np.sum(values)
    numpy_seconds = time.perf_counter() - start

    print(f"Python sum: {python_total:.4f} | Time: {python_seconds:.4f} seconds")
    print(f"NumPy sum:  {numpy_total:.4f} | Time: {numpy_seconds:.4f} seconds")
    if numpy_seconds > 0:
        print(f"NumPy speed ratio: {python_seconds / numpy_seconds:.2f}x (approx.)")


def scrape_articles():
    print("\n[2/6] Scraping the lab URLs...")
    articles = []

    for index, url in enumerate(URLS, start=1):
        print(f"  Fetching article {index}/{len(URLS)}: {url}")
        try:
            response = requests.get(url, headers=HEADERS, timeout=25)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "lxml")

            title_tag = soup.select_one("h3.post-title")
            body_tag = soup.select_one("div.post-body.entry-content")

            title = title_tag.get_text(" ", strip=True) if title_tag else ""
            if body_tag:
                paragraphs = [
                    p.get_text(" ", strip=True)
                    for p in body_tag.find_all("p")
                ]
                body = " ".join(p for p in paragraphs if len(p) > 100)
                if not body:
                    body = body_tag.get_text(" ", strip=True)
            else:
                body = ""

            if not title:
                title_tag = soup.find("title")
                title = title_tag.get_text(" ", strip=True) if title_tag else f"Article {index}"

            if len(body) <= 100:
                print("    WARNING: Article body was missing or too short; skipping.")
                continue

            articles.append({"url": url, "title": title, "body": body})
            print(f"    OK: {title[:75]}")
        except requests.RequestException as exc:
            print(f"    WARNING: Could not download this URL: {exc}")
        except Exception as exc:
            print(f"    WARNING: Could not parse this URL: {exc}")

    if not articles:
        raise RuntimeError(
            "No articles were scraped. Check your internet connection and whether "
            "the source websites are accessible."
        )

    df = pd.DataFrame(articles)
    df.to_csv(DATA_DIR / "scraped_articles.csv", index=False, encoding="utf-8-sig")
    print(f"Saved {len(df)} article(s) to data/scraped_articles.csv")
    return df


def extract_spacy_features(df):
    print("\n[3/6] Extracting spaCy features...")
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "spaCy English model is missing. Run this command in the VS Code "
            "terminal: python -m spacy download en_core_web_sm"
        ) from exc

    token_counts = []
    sentence_counts = []
    entity_counts = []
    noun_counts = []

    for doc in nlp.pipe(df["body"].fillna("").astype(str).tolist(), batch_size=4):
        token_counts.append(sum(1 for token in doc if not token.is_space))
        sentence_counts.append(sum(1 for _ in doc.sents))
        entity_counts.append(len(doc.ents))
        noun_counts.append(sum(1 for token in doc if token.pos_ == "NOUN"))

    result = df.copy()
    result["title_length"] = result["title"].fillna("").str.len()
    result["num_tokens"] = token_counts
    result["num_sentences"] = sentence_counts
    result["num_entities"] = entity_counts
    result["num_nouns"] = noun_counts
    result.to_csv(DATA_DIR / "feature_engineered_articles.csv", index=False, encoding="utf-8-sig")
    return result


def add_sentiment(df):
    print("\n[4/6] Calculating TextBlob sentiment...")
    result = df.copy()
    sentiments = result["body"].fillna("").astype(str).apply(lambda text: TextBlob(text).sentiment)
    result["sentiment_polarity"] = sentiments.apply(lambda value: value.polarity)
    result["sentiment_subjectivity"] = sentiments.apply(lambda value: value.subjectivity)
    result.to_csv(DATA_DIR / "final_dataset.csv", index=False, encoding="utf-8-sig")
    return result


def create_eda_plots(df):
    print("\n[5/6] Creating EDA plots...")
    sns.set_theme()

    plt.figure(figsize=(9, 5))
    sns.histplot(df["title_length"], bins=10, kde=True)
    plt.title("Distribution of Article Title Length")
    plt.xlabel("Title length (characters)")
    plt.ylabel("Number of articles")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "title_distribution.png", dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    sns.histplot(df["sentiment_polarity"], bins=10, kde=True)
    plt.title("Distribution of Sentiment Polarity")
    plt.xlabel("Sentiment polarity")
    plt.ylabel("Number of articles")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_distribution.png", dpi=160)
    plt.close()

    pair_columns = [
        "title_length", "num_tokens", "num_sentences",
        "num_entities", "sentiment_polarity"
    ]
    pair_grid = sns.pairplot(df[pair_columns].dropna(), diag_kind="hist")
    pair_grid.fig.suptitle("Relationships Between Article Features", y=1.02)
    pair_grid.savefig(OUTPUT_DIR / "feature_pairplot.png", dpi=150)
    plt.close("all")

    print("Saved title_distribution.png, sentiment_distribution.png, and feature_pairplot.png")


def create_tfidf_plot(df):
    print("\n[6/6] Calculating TF-IDF top words...")
    corpus = df["body"].fillna("").astype(str).tolist()
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
    matrix = vectorizer.fit_transform(corpus)
    scores = np.asarray(matrix.mean(axis=0)).ravel()
    words = vectorizer.get_feature_names_out()

    top_words = (
        pd.DataFrame({"word": words, "tfidf_score": scores})
        .sort_values("tfidf_score", ascending=False)
    )
    top_words.to_csv(DATA_DIR / "tfidf_top_words.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(11, 6))
    sns.barplot(data=top_words, x="tfidf_score", y="word")
    plt.title("Top 20 Words by Average TF-IDF Score")
    plt.xlabel("Average TF-IDF score")
    plt.ylabel("Word")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tfidf_top_words.png", dpi=160)
    plt.close()
    print("Saved data/tfidf_top_words.csv and outputs/tfidf_top_words.png")


def main():
    print("=" * 72)
    print("LAB 02: WEB SCRAPING, FEATURE ENGINEERING, AND EDA")
    print("=" * 72)
    benchmark_numpy_vs_python()
    articles = scrape_articles()
    featured = extract_spacy_features(articles)
    final_df = add_sentiment(featured)
    create_eda_plots(final_df)
    create_tfidf_plot(final_df)

    print("\n" + "=" * 72)
    print("LAB COMPLETED")
    print(f"Articles processed: {len(final_df)}")
    print(f"Final dataset: {DATA_DIR / 'final_dataset.csv'}")
    print(f"Graphs folder: {OUTPUT_DIR}")
    print("=" * 72)
    print("\nIf fewer than five articles were collected, check the warnings above.")


if __name__ == "__main__":
    main()
