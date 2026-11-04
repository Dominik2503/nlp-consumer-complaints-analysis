
from pathlib import Path
import re
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import nltk
import spacy

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Grundeinstellungen
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "complaints_processed.csv"
RESULTS = ROOT / "results"

N_TOPICS = 5
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

print("Texte für die Analyse:", len(texts))


# --------------------------------------------------
# 5. Vektorisierung: Bag of Words
# --------------------------------------------------

print("\n4. Vektorisierung mit Bag of Words ...")

count_vectorizer = CountVectorizer(
    max_features=MAX_FEATURES,
    min_df=3,
    max_df=0.95,
    ngram_range=(1, 2)
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
    ngram_range=(1, 2),
    sublinear_tf=True
)

tfidf_matrix = tfidf_vectorizer.fit_transform(texts)

print("TF-IDF Matrix:", tfidf_matrix.shape)


# --------------------------------------------------
# 7. Topic Modeling: LDA
# --------------------------------------------------

print("\n6. Trainiere LDA ...")

lda = LatentDirichletAllocation(
    n_components=N_TOPICS,
    random_state=RANDOM_STATE,
    max_iter=10,
    learning_method="batch",
    n_jobs=-1
)

lda.fit(bow_matrix)


# --------------------------------------------------
# 8. Topic Modeling: NMF
# --------------------------------------------------

print("\n7. Trainiere NMF ...")

nmf = NMF(
    n_components=N_TOPICS,
    random_state=RANDOM_STATE,
    init="nndsvda",
    max_iter=400
)

nmf.fit(tfidf_matrix)


# --------------------------------------------------
# 9. Wichtigste Begriffe extrahieren
# --------------------------------------------------

def extract_topics(model, feature_names):

    topics = []

    for index, weights in enumerate(model.components_):

        top_indices = weights.argsort()[-12:][::-1]

        words = [
            feature_names[i]
            for i in top_indices
        ]

        topics.append({
            "Thema": f"Topic {index + 1}",
            "Schlüsselbegriffe": ", ".join(words)
        })

    return pd.DataFrame(topics)


lda_topics = extract_topics(
    lda,
    count_vectorizer.get_feature_names_out()
)

nmf_topics = extract_topics(
    nmf,
    tfidf_vectorizer.get_feature_names_out()
)


# --------------------------------------------------
# 10. Dokumente Themen zuordnen
# --------------------------------------------------

lda_assignments = lda.transform(bow_matrix).argmax(axis=1)
nmf_assignments = nmf.transform(tfidf_matrix).argmax(axis=1)

lda_counts = [
    (lda_assignments == i).sum()
    for i in range(N_TOPICS)
]

nmf_counts = [
    (nmf_assignments == i).sum()
    for i in range(N_TOPICS)
]

lda_topics["Anzahl Beschwerden"] = lda_counts
nmf_topics["Anzahl Beschwerden"] = nmf_counts


# --------------------------------------------------
# 11. Ergebnisse speichern
# --------------------------------------------------

print("\n8. Speichere Ergebnisse ...")

lda_topics.to_csv(
    RESULTS / "lda_topics.csv",
    index=False
)

nmf_topics.to_csv(
    RESULTS / "nmf_topics.csv",
    index=False
)

summary = pd.DataFrame([{
    "Gesamtdatensätze": total_rows,
    "Fehlende Texte": missing,
    "Doppelte Texte": duplicates,
    "Eindeutige Texte nach Bereinigung": len(df),
    "Analysierte Texte": sample_size,
    "Anzahl Themen": N_TOPICS
}])

summary.to_csv(
    RESULTS / "dataset_summary.csv",
    index=False
)


# --------------------------------------------------
# 12. Visualisierung
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
        dpi=160
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
# 13. Ergebnisse anzeigen
# --------------------------------------------------

print("\n===================================")
print("ANALYSE ERFOLGREICH ABGESCHLOSSEN")
print("===================================")

print("\nLDA – Ergebnisse:")
print(lda_topics.to_string(index=False))

print("\nNMF – Ergebnisse:")
print(nmf_topics.to_string(index=False))

print("\nAlle Ergebnisse gespeichert unter:")
print(RESULTS)
