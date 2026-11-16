from pathlib import Path
import re
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import nltk
import spacy

from nltk.corpus import stopwords
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Grundeinstellungen
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "complaints_processed.csv"
RESULTS = ROOT / "results"

TOPIC_RANGE = range(3, 11)
COHERENCE_TOP_WORDS = 10
SAMPLE_SIZE = 20000
MAX_FEATURES = 5000
RANDOM_STATE = 42

RESULTS.mkdir(exist_ok=True)


# --------------------------------------------------
# 2. Daten laden
# --------------------------------------------------

print("1. Lade Datensatz ...")

df = pd.read_csv(DATA_PATH)

print("Anzahl Datensätze:", len(df))
print("Vorhandene Spalten:", df.columns.tolist())

if "narrative" not in df.columns:
    raise ValueError("Die Spalte 'narrative' wurde nicht gefunden.")

total_rows = len(df)
missing = df["narrative"].isna().sum()

df = df.dropna(subset=["narrative"]).copy()
df["narrative"] = df["narrative"].astype(str)

duplicates = df["narrative"].duplicated().sum()

print("Fehlende Texte:", missing)
print("Doppelte Texte:", duplicates)


# --------------------------------------------------
# 3. Textvorverarbeitung
# --------------------------------------------------

print("\n2. Bereinige Beschwerdetexte ...")

nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))

# Englischer spaCy-Tokenizer
nlp = spacy.blank("en")


def clean_text(text):

    text = text.lower()

    # URLs entfernen
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Zahlen und Sonderzeichen entfernen
    text = re.sub(r"[^a-z\s]", " ", text)

    # Mehrfache Leerzeichen entfernen
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenisierung und Stopword-Entfernung
    tokens = []

    for token in nlp.make_doc(text):
        word = token.text

        if (
            word.isalpha()
            and len(word) > 2
            and word not in stop_words
        ):
            tokens.append(word)

    return " ".join(tokens)


# Exakte Duplikate entfernen
df = df.drop_duplicates(subset=["narrative"]).copy()

df["clean_text"] = df["narrative"].apply(clean_text)

# Leere Texte entfernen
df = df[df["clean_text"].str.len() > 0].copy()

print("Verwertbare eindeutige Texte:", len(df))


# --------------------------------------------------
# 4. Stichprobe erstellen
# --------------------------------------------------

print("\n3. Erstelle Stichprobe ...")

sample_size = min(SAMPLE_SIZE, len(df))

sample = df.sample(
    n=sample_size,
    random_state=RANDOM_STATE
)

texts = sample["clean_text"].tolist()
# Texte für Gensim tokenisieren
tokenized_documents = [
    text.split() for text in texts
]

# Gensim-Wörterbuch erstellen
dictionary = Dictionary(tokenized_documents)

dictionary.filter_extremes(
    no_below=3,
    no_above=0.95
)

print("Texte für die Analyse:", len(texts))


# --------------------------------------------------
# 5. Vektorisierung: Bag of Words
# --------------------------------------------------

print("\n4. Vektorisierung mit Bag of Words ...")

count_vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    min_df=3,
    max_df=0.95,
    ngram_range=(1, 1)
)

bow_matrix = count_vectorizer.fit_transform(texts)

print("Bag of Words Matrix:", bow_matrix.shape)


# --------------------------------------------------
# 6. Vektorisierung: TF-IDF
# --------------------------------------------------

print("\n5. Vektorisierung mit TF-IDF ...")

tfidf_vectorizer = TfidfVectorizer(
    max_features=MAX_FEATURES,
    min_df=3,
    max_df=0.95,
    ngram_range=(1, 1),
    sublinear_tf=True
)

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

print("TF-IDF Matrix:", tfidf_matrix.shape)


# --------------------------------------------------
# 7. Hilfsfunktionen für Topic Modeling
# --------------------------------------------------

def extract_topic_words(model, feature_names, number_of_words=12):

    topics = []

    for weights in model.components_:
        top_indices = weights.argsort()[-number_of_words:][::-1]

        words = [
            feature_names[i]
            for i in top_indices
        ]

        topics.append(words)

    return topics


def calculate_coherence(topics):

    valid_topics = []

    for topic in topics:
        words = [
            word for word in topic
            if word in dictionary.token2id
        ]

        if len(words) >= 2:
            valid_topics.append(words)

    if not valid_topics:
        return 0.0

    coherence_model = CoherenceModel(
        topics=valid_topics,
        texts=tokenized_documents,
        dictionary=dictionary,
        coherence="c_v",
        processes=1
    )

    return coherence_model.get_coherence()


# --------------------------------------------------
# 8. Optimale Topic-Anzahl ermitteln
# --------------------------------------------------

print("\n6. Teste verschiedene Topic-Anzahlen ...")

lda_features = count_vectorizer.get_feature_names_out()
nmf_features = tfidf_vectorizer.get_feature_names_out()

evaluation_results = []

for n_topics in TOPIC_RANGE:

    print(f"\nTeste {n_topics} Topics ...")

    # LDA trainieren
    lda_test = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        max_iter=10,
        learning_method="batch",
        n_jobs=-1
    )

    lda_test.fit(bow_matrix)

    lda_words = extract_topic_words(
        lda_test,
        lda_features,
        COHERENCE_TOP_WORDS
    )

    lda_score = calculate_coherence(lda_words)

    # NMF trainieren
    nmf_test = NMF(
        n_components=n_topics,
        random_state=RANDOM_STATE,
        init="nndsvda",
        max_iter=400
    )

    nmf_test.fit(tfidf_matrix)

    nmf_words = extract_topic_words(
        nmf_test,
        nmf_features,
        COHERENCE_TOP_WORDS
    )

    nmf_score = calculate_coherence(nmf_words)

    evaluation_results.append({
        "Anzahl Topics": n_topics,
        "LDA C_v": lda_score,
        "NMF C_v": nmf_score
    })

    print(f"LDA C_v: {lda_score:.4f}")
    print(f"NMF C_v: {nmf_score:.4f}")


# Kohärenzwerte speichern
evaluation_df = pd.DataFrame(evaluation_results)

evaluation_df.to_csv(
    RESULTS / "coherence_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# 9. Beste Topic-Anzahl auswählen
# --------------------------------------------------

best_lda_row = evaluation_df.loc[
    evaluation_df["LDA C_v"].idxmax()
]

best_nmf_row = evaluation_df.loc[
    evaluation_df["NMF C_v"].idxmax()
]

best_lda_topics = int(best_lda_row["Anzahl Topics"])
best_nmf_topics = int(best_nmf_row["Anzahl Topics"])

print("\nOptimale Topic-Anzahlen:")
print("LDA:", best_lda_topics)
print("NMF:", best_nmf_topics)


# --------------------------------------------------
# 10. Finale Modelle trainieren
# --------------------------------------------------

print("\n7. Trainiere finale Modelle ...")

lda = LatentDirichletAllocation(
    n_components=best_lda_topics,
    random_state=RANDOM_STATE,
    max_iter=10,
    learning_method="batch",
    n_jobs=-1
)

lda.fit(bow_matrix)

nmf = NMF(
    n_components=best_nmf_topics,
    random_state=RANDOM_STATE,
    init="nndsvda",
    max_iter=400
)

nmf.fit(tfidf_matrix)


# --------------------------------------------------
# 11. Finale Topics und Verteilungen erstellen
# --------------------------------------------------

lda_topic_words = extract_topic_words(
    lda,
    lda_features,
    12
)

nmf_topic_words = extract_topic_words(
    nmf,
    nmf_features,
    12
)


def create_topic_table(model, topic_words, matrix):

    assignments = model.transform(matrix).argmax(axis=1)

    topic_counts = [
        int((assignments == i).sum())
        for i in range(model.n_components)
    ]

    results = []

    for i, words in enumerate(topic_words):
        results.append({
            "Thema": f"Topic {i + 1}",
            "Schlüsselbegriffe": ", ".join(words),
            "Anzahl Beschwerden": topic_counts[i]
        })

    return pd.DataFrame(results)


lda_topics = create_topic_table(
    lda,
    lda_topic_words,
    bow_matrix
)

nmf_topics = create_topic_table(
    nmf,
    nmf_topic_words,
    tfidf_matrix
)


# --------------------------------------------------
# 12. Ergebnisse speichern
# --------------------------------------------------

print("\n8. Speichere Ergebnisse ...")

lda_topics.to_csv(
    RESULTS / "lda_topics.csv",
    index=False,
    encoding="utf-8-sig"
)

nmf_topics.to_csv(
    RESULTS / "nmf_topics.csv",
    index=False,
    encoding="utf-8-sig"
)

summary = pd.DataFrame([{
    "Gesamtdatensätze": total_rows,
    "Fehlende Texte": missing,
    "Doppelte Texte": duplicates,
    "Eindeutige Texte nach Bereinigung": len(df),
    "Analysierte Texte": len(texts),
    "Optimale LDA Topic-Anzahl": best_lda_topics,
    "Optimale NMF Topic-Anzahl": best_nmf_topics,
    "LDA C_v": float(best_lda_row["LDA C_v"]),
    "NMF C_v": float(best_nmf_row["NMF C_v"])
}])

summary.to_csv(
    RESULTS / "dataset_summary.csv",
    index=False,
    encoding="utf-8-sig"
)


# --------------------------------------------------
# 13. Kohärenz visualisieren
# --------------------------------------------------

plt.figure(figsize=(9, 5))

plt.plot(
    evaluation_df["Anzahl Topics"],
    evaluation_df["LDA C_v"],
    marker="o",
    label="LDA"
)

plt.plot(
    evaluation_df["Anzahl Topics"],
    evaluation_df["NMF C_v"],
    marker="o",
    label="NMF"
)

plt.xlabel("Anzahl Topics")
plt.ylabel("C_v-Kohärenz")
plt.title("Vergleich der Topic-Kohärenz")
plt.xticks(list(TOPIC_RANGE))
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "coherence_comparison.png",
    dpi=300
)

plt.close()


# --------------------------------------------------
# 14. Topic-Verteilungen visualisieren
# --------------------------------------------------

def create_chart(data, title, filename):

    plt.figure(figsize=(9, 5))

    plt.bar(
        data["Thema"],
        data["Anzahl Beschwerden"]
    )

    plt.title(title)
    plt.xlabel("Thema")
    plt.ylabel("Anzahl Beschwerden")

    plt.tight_layout()

    plt.savefig(
        RESULTS / filename,
        dpi=300
    )

    plt.close()


create_chart(
    lda_topics,
    "Themenverteilung – LDA",
    "lda_topic_distribution.png"
)

create_chart(
    nmf_topics,
    "Themenverteilung – NMF",
    "nmf_topic_distribution.png"
)


# --------------------------------------------------
# 15. Ergebnisse anzeigen
# --------------------------------------------------

print("\n===================================")
print("ANALYSE ERFOLGREICH ABGESCHLOSSEN")
print("===================================")

print("\nLDA – Ergebnisse:")
print(lda_topics.to_string(index=False))

print("\nNMF – Ergebnisse:")
print(nmf_topics.to_string(index=False))

print("\nKohärenzvergleich:")
print(evaluation_df.to_string(index=False))

print("\nAlle Ergebnisse gespeichert unter:")
print(RESULTS)

