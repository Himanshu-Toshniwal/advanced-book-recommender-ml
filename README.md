# 📚 Advanced Book Recommender System

A machine learning-powered book recommendation system built with Python, using the real **Book-Crossing dataset from Kaggle** (270k+ books, 1M+ ratings). Features content-based filtering, collaborative filtering, EDA visualizations, and a complete Docker + CI/CD pipeline.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange)
![matplotlib](https://img.shields.io/badge/matplotlib-3.7.2-green)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-black)

---

## �️ Pr/oject Structure

```
advanced-book-recommender-ml/
├── src/
│   ├── main.py              # Core app – recommender + CLI menu + charts
│   └── download_data.py     # Kaggle dataset downloader / sample data generator
├── data/
│   ├── BX-Books.csv         # Book metadata (ISBN, title, author, year, publisher)
│   ├── BX-Ratings.csv       # User ratings (1–10 scale)
│   └── BX-Users.csv         # User demographics
├── .github/
│   └── workflows/
│       └── docker-build-deploy.yml   # CI/CD pipeline
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

## 🤖 ML Algorithms

| Algorithm | How it works |
|-----------|-------------|
| TF-IDF + Cosine Similarity | Vectorizes book title, author, publisher — finds similar books by content |
| Item-based Collaborative Filtering | Finds users with similar taste, recommends what they liked |
| Weighted Popularity Score | Bayesian average rating — balances avg rating with number of ratings |

---

## 📊 Dataset — Book-Crossing (Kaggle)

- **Source**: [kaggle.com/datasets/arashnic/book-recommendation-dataset](https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset)
- 271,379 books | 278,858 users | 1,149,780 ratings
- Ratings on a scale of 1–10 (0 = implicit feedback, excluded)

---

## 🚀 Quick Start

### Option 1 — Local Python

```bash
# Install dependencies
pip install -r requirements.txt

# Setup dataset (first time only)
python src/download_data.py
# → Has Kaggle API key? y = auto-download real data
# → No key? n = generates realistic sample data automatically

# Run the app
python src/main.py
```

### Option 2 — Docker

```bash
# Build
docker-compose up --build

# Run interactively
docker run -it --rm \
  -v "$(pwd)/data:/app/data" \
  advanced-book-recommender-ml-book-recommender:latest
```

> On Windows:
> ```cmd
> docker run -it --rm -v "D:\path\to\project\data:/app/data" advanced-book-recommender-ml-book-recommender:latest
> ```

---

## 🖥️ App Menu

```
============================================================
  📚  BOOK RECOMMENDER  –  MAIN MENU
============================================================
  1.  🔍  Search Books (title / author / publisher)
  2.  🎯  Content-Based Recommendations (by ISBN)
  3.  👥  Collaborative Recommendations (by User-ID)
  4.  🏆  Popular Books
  ──────────────────────────────────────────────────────────
  5.  📊  [CHART] Top Books by Popularity
  6.  📊  [CHART] Similarity Heatmap
  7.  📊  [CHART] Avg Rating vs Rating Count
  8.  📊  [CHART] Year Trend (Rating & Volume)
  9.  📊  [CHART] Re-run EDA Charts
  ──────────────────────────────────────────────────────────
 10.  ❌  Exit
```

### EDA Charts (shown at startup)
- Rating distribution histogram
- Top 15 most rated authors
- Books published per decade
- Top publishers pie chart
- User activity distribution

---

## � Sample ISBiNs to Try

| ISBN | Book |
|------|------|
| `0743247531` | The Da Vinci Code |
| `0385737951` | The Hunger Games |
| `0618640150` | The Lord of the Rings |
| `0060931647` | Harry Potter |
| `0451524934` | 1984 |
| `0735224293` | Atomic Habits |
| `0743273516` | To Kill a Mockingbird |

---

## 🔄 CI/CD Pipeline (GitHub Actions)

```
Push to main
    │
    ▼
🧪 Test Job
    ├── Install dependencies
    ├── Generate sample dataset
    └── Run pytest (imports, data loader, recommender, search)
    │
    ▼
🐳 Build & Push (main branch only)
    ├── Docker Buildx setup
    ├── Login to DockerHub
    └── Build + push :latest and :<sha> tags
    │
    ▼
🚀 Deploy
    └── Confirm image published
```

### Required GitHub Secrets

| Secret | Value |
|--------|-------|
| `DOCKERHUB_USERNAME` | Your DockerHub username |
| `DOCKERHUB_TOKEN` | DockerHub → Account Settings → Security → New Access Token |

---

## 🛠️ Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | 2.0.3 | Data loading and manipulation |
| numpy | 1.24.3 | Numerical operations |
| scikit-learn | 1.3.0 | TF-IDF, cosine similarity, SVD |
| matplotlib | 3.7.2 | EDA charts and visualizations |

---

## Prerequisites

- Python 3.10+
- Docker (optional)
- Kaggle account (optional — sample data works without it)
