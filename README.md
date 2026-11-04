# NLP-gestützte Analyse von Beschwerdetexten

## Projektziel
Dieses Projekt untersucht englischsprachige Kundenbeschwerden mit NLP, um wiederkehrende Themen zu identifizieren. Die Daten stammen aus dem Kaggle-Datensatz „Consumer Complaints Dataset“. Sie dienen als Beispieldaten und sind nicht repräsentativ für die Bevölkerung einer Kommune.

## Projektstruktur
- `data/`: Hier die Datei `complaints_processed.csv` ablegen.
- `src/analyze_complaints.py`: vollständige Analysepipeline.
- `results/`: automatisch erzeugte Tabellen und Diagramme.

## Einrichtung unter Windows mit VS Code

1. Python 3.11 oder 3.12 installieren und bei der Installation „Add Python to PATH“ aktivieren.
2. Diesen Projektordner in VS Code öffnen.
3. In VS Code ein Terminal öffnen: **Terminal > New Terminal**.
4. Virtuelle Umgebung erstellen:

```powershell
py -m venv .venv
```

5. Virtuelle Umgebung aktivieren:

```powershell
.\.venv\Scripts\Activate.ps1
```

Falls PowerShell die Aktivierung blockiert, einmalig für das aktuelle Terminal ausführen:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

6. Abhängigkeiten installieren:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

7. Datensatz in `data/complaints_processed.csv` speichern.
8. Analyse starten:

```powershell
python src/analyze_complaints.py
```

Die Ergebnisse werden im Ordner `results/` gespeichert.

## Methodik
- Vorverarbeitung: pandas, NLTK und spaCy (Tokenizer)
- Vektorisierung: CountVectorizer (Bag of Words) und TfidfVectorizer
- Themenextraktion: LDA auf Bag-of-Words-Matrix und NMF auf TF-IDF-Matrix
- Visualisierung: Matplotlib

## Methodische Einschränkung
Die ausgewählte CSV-Datei ist bereits stark vorverarbeitet: Kleinbuchstaben und entfernte Satzzeichen sind erkennbar; zahlreiche Wörter scheinen bereits gestemmt zu sein. Deshalb wird keine erneute Lemmatisierung durchgeführt, da sich bereits gestemmte Wortformen nicht zuverlässig zurück in Grundformen überführen lassen. Die Bereinigung beschränkt sich auf eine kontrollierte Normalisierung, Tokenisierung und Stopwortentfernung.

Zur Begrenzung der Rechenzeit werden die Modelle standardmäßig auf einer reproduzierbaren Zufallsstichprobe von bis zu 20.000 eindeutigen Texten trainiert. Die Anzahl der Themen ist zunächst auf fünf festgelegt. Die Stichprobengröße und Themenanzahl können im Skript angepasst werden.

## Datenschutz und Reproduzierbarkeit
Der Datensatz enthält Beschwerdetexte. Vor einer Veröffentlichung ist zu prüfen, ob darin personenbezogene oder sensible Angaben vorkommen. Die Rohdaten werden daher nicht automatisch in GitHub aufgenommen. Die Zufallsstichprobe ist durch `RANDOM_STATE = 42` reproduzierbar.
