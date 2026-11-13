Ziel des Projekts ist es, wiederkehrende Themen in unstrukturierten Verbraucherbeschwerden mithilfe von Natural Language Processing (NLP) zu erkennen. 
Als Datengrundlage dient der öffentlich verfügbare Kaggle-Datensatz „Consumer Complaints Dataset for NLP“ . Er enthält englischsprachige Beschwerden aus dem Finanzbereich und eignet sich zur Erprobung der Verfahren. Die Ergebnisse sind jedoch nicht ohne Weiteres auf kommunale Beschwerdetexte übertragbar.

Die Umsetzung erfolgt mit Python. Zunächst werden die Daten mit pandas eingelesen und doppelte Einträge gelöscht. Anschließend werden die Texte vereinheitlicht: URLs, Zahlen und Sonderzeichen
werden entfernt, die Texte kleingeschrieben und mehrfach vorkommende Leerzeichen bereinigt. Mit spaCy erfolgt die Tokenisierung; englische Stoppwörter werden mithilfe von NLTK entfernt. Die Analyse basiert auf einer reproduzierbaren Stichprobe von maximal 20.000 Texten.

Für die Umwandlung der Texte in numerische Merkmale werden zwei Verfahren verglichen: Bag-of-Words (CountVectorizer) und TF-IDF (TfidfVectorizer). Anschließend werden zwei Topic-Modeling-Verfahren eingesetzt: Latent Dirichlet Allocation (LDA) auf der Bag-of-Words-Matrix und Non-negative Matrix Factorization (NMF) auf der TF-IDF-Matrix. Die Untersuchung wird bewusst auf fünf Topics pro Verfahren begrenzt, um die Themen übersichtlich darzustellen und die Ergebnisse von LDA und NMF vergleichbar auswerten zu können. Diese Begrenzung ist eine methodische Festlegung des Projekts; die optimale Topic-Anzahl wird nicht automatisch ermittelt.

Für jede Topic werden die wichtigsten Schlüsselbegriffe extrahiert und die Beschwerden dem jeweils
dominanten Topic zugeordnet. Die Ergebnisse werden als CSV-Dateien gespeichert und durch
Balkendiagramme zur Verteilung der Beschwerden auf die Topics visualisiert. Die Themen werden anschließend inhaltlich interpretiert. Die Implementierung verwendet Tokenisierung und Stoppwortentfernung, aber keine separate Lemmatisierung. 
