"""
Task 4: NLP Sentiment Classification Pipeline (TF-IDF)
Curriculum Reference: DT-2026-NKPSG
Focus: Text Preprocessing, TF-IDF Feature Extraction, Naive Bayes & Logistic Regression

This script provides end-to-end functionality to:
1. Ingest customer review records from CSV.
2. Clean and tokenize text (HTML removal, punctuation stripping, lowercasing, stopword filtering).
3. Transform text into numerical TF-IDF feature matrices (max_features=5000, unigrams + bigrams).
4. Train and benchmark Multinomial Naive Bayes and Logistic Regression classifiers.
5. Export evaluation figures to assets/ directory.
6. Provide an interactive predict_sentiment() helper for live text inference.
"""

import os
import re
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

# Standard English stop words
ENGLISH_STOP_WORDS = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "once",
    "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't",
    "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that",
    "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who",
    "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll",
    "you're", "you've", "your", "yours", "yourself", "yourselves"
])


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """
    Cleans raw review text through regex normalization:
    1. Removes HTML markup tags.
    2. Removes URL patterns.
    3. Normalizes text to lowercase.
    4. Removes punctuation and special symbols.
    5. Strips extra whitespace.
    6. Filters out standard stopwords if enabled.
    """
    if not isinstance(text, str):
        return ""

    # Strip HTML tags
    cleaned = re.sub(r"<[^>]+>", " ", text)

    # Strip URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)

    # Strip punctuation and numbers
    cleaned = re.sub(r"[^\w\s]", " ", cleaned)
    cleaned = re.sub(r"\d+", " ", cleaned)

    # Convert to lowercase and tokenize on whitespace
    tokens = cleaned.lower().split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]

    return " ".join(tokens)


def load_and_preprocess_data(data_path: str, filter_binary: bool = True) -> pd.DataFrame:
    """
    Loads review dataset, applies cleaning, and encodes sentiment labels.
    """
    df = pd.read_csv(data_path)

    # Filter to binary classes if requested
    if filter_binary:
        df = df[df["sentiment"].isin(["positive", "negative"])].copy()

    df["cleaned_text"] = df["review_text"].apply(clean_text)
    df["label"] = df["sentiment"].map({"negative": 0, "positive": 1})
    df["word_count"] = df["cleaned_text"].apply(lambda t: len(t.split()))

    return df


def build_tfidf_features(
    X_train: pd.Series,
    X_test: pd.Series,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2)
):
    """
    Fits TfidfVectorizer on training text and transforms both train and test splits.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return vectorizer, X_train_tfidf, X_test_tfidf


def train_models(X_train, y_train):
    """
    Fits Multinomial Naive Bayes and Logistic Regression classifiers.
    """
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train, y_train)

    lr_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    return {"MultinomialNB": nb_model, "LogisticRegression": lr_model}


def evaluate_models(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    """
    Evaluates models across standard classification metrics and 5-fold cross-validation.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    records = []

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        cv_acc = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
        cv_auc = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc").mean()

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        records.append({
            "Model": name,
            "CV Accuracy": round(cv_acc, 4),
            "CV ROC-AUC": round(cv_auc, 4),
            "Test Accuracy": round(acc, 4),
            "Test Precision": round(prec, 4),
            "Test Recall": round(rec, 4),
            "Test F1-Score": round(f1, 4),
            "Test ROC-AUC": round(auc, 4)
        })

    return pd.DataFrame(records)


def predict_sentiment(text: str, model, vectorizer) -> dict:
    """
    Inference helper: takes raw custom user input text, preprocesses it,
    transforms via TF-IDF, and returns predicted class and probability score.
    """
    cleaned = clean_text(text)
    vec = vectorizer.transform([cleaned])
    pred_class = model.predict(vec)[0]
    pred_prob = model.predict_proba(vec)[0]

    sentiment_label = "positive" if pred_class == 1 else "negative"
    confidence = float(pred_prob[pred_class])

    return {
        "raw_text": text,
        "cleaned_text": cleaned,
        "predicted_sentiment": sentiment_label,
        "confidence": round(confidence, 4),
        "negative_prob": round(float(pred_prob[0]), 4),
        "positive_prob": round(float(pred_prob[1]), 4)
    }


def generate_visualizations(df: pd.DataFrame, models: dict, vectorizer, X_test, y_test, output_dir: str):
    """
    Generates and saves the required evaluation plots to assets/.
    """
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Figure 1: Class distribution and review word lengths
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    palette = {"positive": "#2ecc71", "negative": "#e74c3c"}
    sns.countplot(
        data=df,
        x="sentiment",
        ax=axes[0],
        palette=palette,
        hue="sentiment",
        legend=False
    )
    axes[0].set_title("Corpus Sentiment Class Distribution", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Sentiment Class", fontsize=11)
    axes[0].set_ylabel("Review Count", fontsize=11)
    for p in axes[0].patches:
        height = int(p.get_height())
        axes[0].annotate(f"{height}", (p.get_x() + p.get_width() / 2.0, height + 1),
                         ha="center", va="bottom", fontsize=11, fontweight="bold")

    sns.kdeplot(
        data=df,
        x="word_count",
        hue="sentiment",
        palette=palette,
        fill=True,
        common_norm=False,
        ax=axes[1],
        alpha=0.4
    )
    axes[1].set_title("Preprocessed Word Count Distribution by Class", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Cleaned Word Count per Review", fontsize=11)
    axes[1].set_ylabel("Density", fontsize=11)

    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "01_class_distribution_and_lengths.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()

    # Figure 2: Side-by-side Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    class_labels = ["Negative", "Positive"]

    for i, (name, model) in enumerate(models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cm_perc = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

        annot = np.empty_like(cm, dtype=object)
        for r in range(cm.shape[0]):
            for c in range(cm.shape[1]):
                annot[r, c] = f"{cm[r, c]}\n({cm_perc[r, c]:.1%})"

        cmap = "Blues" if name == "MultinomialNB" else "Greens"
        sns.heatmap(cm, annot=annot, fmt="", cmap=cmap, cbar=False,
                    xticklabels=class_labels, yticklabels=class_labels,
                    ax=axes[i], annot_kws={"size": 12, "weight": "bold"})
        axes[i].set_title(f"{name} Confusion Matrix", fontsize=13, fontweight="bold", pad=10)
        axes[i].set_xlabel("Predicted Class", fontsize=11)
        axes[i].set_ylabel("True Class", fontsize=11)

    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "02_confusion_matrices.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()

    # Figure 3: ROC and Precision-Recall Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"MultinomialNB": "#2980b9", "LogisticRegression": "#27ae60"}
    for name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=colors[name], lw=2.2)

        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        axes[1].plot(rec, prec, label=f"{name}", color=colors[name], lw=2.2)

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.2, alpha=0.7)
    axes[0].set_title("Receiver Operating Characteristic (ROC) Curve", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("False Positive Rate", fontsize=11)
    axes[0].set_ylabel("True Positive Rate", fontsize=11)
    axes[0].legend(loc="lower right", frameon=True)

    axes[1].set_title("Precision-Recall Curve", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Recall", fontsize=11)
    axes[1].set_ylabel("Precision", fontsize=11)
    axes[1].legend(loc="lower left", frameon=True)

    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "03_roc_curves.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()

    # Figure 4: Top TF-IDF Features by Sentiment
    lr_model = models["LogisticRegression"]
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = lr_model.coef_[0]

    top_pos_idx = np.argsort(coefs)[-12:]
    top_neg_idx = np.argsort(coefs)[:12]

    top_features = pd.DataFrame({
        "Feature": np.concatenate([feature_names[top_neg_idx], feature_names[top_pos_idx]]),
        "Coefficient": np.concatenate([coefs[top_neg_idx], coefs[top_pos_idx]]),
        "Sentiment": ["Negative"] * len(top_neg_idx) + ["Positive"] * len(top_pos_idx)
    })
    top_features = top_features.sort_values(by="Coefficient")

    plt.figure(figsize=(12, 6))
    bar_palette = {"Positive": "#2ecc71", "Negative": "#e74c3c"}
    sns.barplot(
        data=top_features,
        y="Feature",
        x="Coefficient",
        hue="Sentiment",
        palette=bar_palette,
        dodge=False
    )
    plt.title("Top Most Influential TF-IDF N-grams by Polarity (Logistic Regression)",
              fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Log-Odds Coefficient Weight", fontsize=11)
    plt.ylabel("N-Gram Feature", fontsize=11)
    plt.axvline(0, color="black", linestyle="--", alpha=0.6, lw=1)
    plt.tight_layout()

    fig4_path = os.path.join(output_dir, "04_top_tfidf_features.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()

    print(f"Figures exported to {output_dir}")


def run_pipeline():
    """
    Executes the entire NLP sentiment classification workflow.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    expanded_data_path = os.path.join(base_dir, "data", "sentiment_reviews_expanded.csv")
    starter_data_path = os.path.join(base_dir, "data", "sentiment_reviews.csv")
    assets_dir = os.path.join(base_dir, "assets")

    # Use the expanded benchmark dataset if available, fallback to starter file
    data_path = expanded_data_path if os.path.exists(expanded_data_path) else starter_data_path
    print(f"Loading review data from: {data_path}")

    df = load_and_preprocess_data(data_path, filter_binary=True)
    print(f"Preprocessed corpus shape: {df.shape}")
    print(f"Class distribution:\n{df['sentiment'].value_counts()}\n")

    # Train/test split (80/20 stratified)
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["label"],
        test_size=0.20,
        random_state=42,
        stratify=df["label"]
    )

    # TF-IDF Vectorization
    vectorizer, X_train, X_test = build_tfidf_features(
        X_train_raw,
        X_test_raw,
        max_features=5000,
        ngram_range=(1, 2)
    )
    print(f"TF-IDF vocabulary size: {len(vectorizer.vocabulary_)} features")

    # Model training
    models = train_models(X_train, y_train)

    # Benchmark evaluation
    metrics_df = evaluate_models(models, X_train, y_train, X_test, y_test)
    print("\nModel Benchmark Summary:")
    print(metrics_df.to_string(index=False))

    # Generate visual assets
    generate_visualizations(df, models, vectorizer, X_test, y_test, assets_dir)

    # Test interactive helper with live custom phrases
    test_phrases = [
        "The build quality is exceptional, exceeded all my expectations!",
        "Terrible customer service and the item arrived completely damaged.",
        "Battery holds charge for days and display is remarkably sharp.",
        "Stopped working after two days, completely defective piece of junk.",
        "Fast shipping, highly recommend this to anyone looking for a reliable tool."
    ]

    print("\nLive Inference Verification with predict_sentiment():")
    for phrase in test_phrases:
        result = predict_sentiment(phrase, models["LogisticRegression"], vectorizer)
        print(f"Text: '{result['raw_text']}'")
        print(f"  -> Predicted: {result['predicted_sentiment'].upper()} (Confidence: {result['confidence']:.2%})\n")


if __name__ == "__main__":
    run_pipeline()
