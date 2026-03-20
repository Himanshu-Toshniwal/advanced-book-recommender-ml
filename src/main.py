"""
Advanced Book Recommender System
Dataset : Kaggle – Book-Crossing (arashnic/book-recommendation-dataset)
Run     : python src/download_data.py   (first time)
          python src/main.py
"""

import os, sys
import pandas as pd
import numpy as np
import matplotlib
import os
# Use TkAgg locally (interactive), Agg in Docker/CI (no display)
if not os.environ.get('MPLBACKEND'):
    try:
        matplotlib.use('TkAgg')
    except Exception:
        matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler
import sklearn

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')

# ─────────────────────────────────────────────────────────────────────────────
#  STARTUP BANNER
# ─────────────────────────────────────────────────────────────────────────────
def print_banner():
    print("\n" + "█"*62)
    print("█" + " "*60 + "█")
    print("█   📚  ADVANCED BOOK RECOMMENDER  v3.0  (Kaggle Data)   █")
    print("█" + " "*60 + "█")
    print(f"█   🔬 scikit-learn  {sklearn.__version__:<8}  📊 matplotlib {matplotlib.__version__:<8}  █")
    print(f"█   🐼 pandas        {pd.__version__:<8}  🔢 numpy      {np.__version__:<8}  █")
    print("█" + " "*60 + "█")
    print("█"*62 + "\n")

# ─────────────────────────────────────────────────────────────────────────────
#  DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
class DataLoader:
    def __init__(self, data_dir=DATA_DIR):
        self.data_dir = data_dir

    def _path(self, fname):
        return os.path.join(self.data_dir, fname)

    def load(self):
        if not all(os.path.exists(self._path(f))
                   for f in ['BX-Books.csv', 'BX-Ratings.csv', 'BX-Users.csv']):
            print("⚠️  Dataset not found. Running downloader...")
            import subprocess
            subprocess.run([sys.executable,
                            os.path.join(os.path.dirname(__file__), 'download_data.py')])

        print("📂 Loading dataset...")
        books   = pd.read_csv(self._path('BX-Books.csv'),   sep=';', on_bad_lines='skip',
                               encoding='latin-1', low_memory=False)
        ratings = pd.read_csv(self._path('BX-Ratings.csv'), sep=';', on_bad_lines='skip',
                               encoding='latin-1', low_memory=False)
        users   = pd.read_csv(self._path('BX-Users.csv'),   sep=';', on_bad_lines='skip',
                               encoding='latin-1', low_memory=False)

        # normalise column names
        books.columns   = [c.strip().replace('"','') for c in books.columns]
        ratings.columns = [c.strip().replace('"','') for c in ratings.columns]
        users.columns   = [c.strip().replace('"','') for c in users.columns]

        # strip stray quotes from values
        for col in books.select_dtypes('object').columns:
            books[col] = books[col].astype(str).str.strip('"').str.strip()

        books['Year-Of-Publication'] = pd.to_numeric(
            books['Year-Of-Publication'], errors='coerce')
        books = books[(books['Year-Of-Publication'] >= 1800) &
                      (books['Year-Of-Publication'] <= 2024)]

        ratings['Book-Rating'] = pd.to_numeric(ratings['Book-Rating'], errors='coerce')
        ratings = ratings[ratings['Book-Rating'] > 0]   # drop implicit 0s

        print(f"   📚 Books   : {len(books):,}")
        print(f"   👤 Users   : {len(users):,}")
        print(f"   ⭐ Ratings : {len(ratings):,}")
        return books, ratings, users


# ─────────────────────────────────────────────────────────────────────────────
#  EDA  (Exploratory Data Analysis)  – shown once at startup
# ─────────────────────────────────────────────────────────────────────────────
class EDA:
    def __init__(self, books, ratings, users):
        self.books   = books
        self.ratings = ratings
        self.users   = users

    def summary(self):
        print("\n" + "="*62)
        print("  📊  DATASET ANALYSIS SUMMARY")
        print("="*62)

        # merge for analysis
        merged = self.ratings.merge(
            self.books[['ISBN','Book-Title','Book-Author','Year-Of-Publication','Publisher']],
            on='ISBN', how='inner')

        print(f"\n  Total Books        : {self.books['ISBN'].nunique():>8,}")
        print(f"  Total Users        : {self.users['User-ID'].nunique():>8,}")
        print(f"  Total Ratings      : {len(self.ratings):>8,}")
        print(f"  Avg Rating         : {self.ratings['Book-Rating'].mean():>8.2f} / 10")
        print(f"  Most Rated Book    : {merged.groupby('Book-Title').size().idxmax()[:45]}")
        print(f"  Most Active Author : {merged.groupby('Book-Author').size().idxmax()[:45]}")

        top_pub = merged.groupby('Publisher').size().nlargest(1).index[0]
        print(f"  Top Publisher      : {str(top_pub)[:45]}")

        yr = self.books['Year-Of-Publication'].dropna()
        print(f"  Publication Range  : {int(yr.min())} – {int(yr.max())}")
        print("="*62)

    def run_all_charts(self):
        """Show all EDA charts at startup."""
        self._plot_rating_distribution()
        self._plot_top_authors()
        self._plot_publications_per_decade()
        self._plot_top_publishers()
        self._plot_user_activity()

    def _plot_rating_distribution(self):
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        fig.suptitle('⭐ Rating Distribution (Book-Crossing Dataset)', fontsize=14, fontweight='bold')

        # histogram
        axes[0].hist(self.ratings['Book-Rating'], bins=10, range=(1,10),
                     color='#3498db', edgecolor='white', rwidth=0.85)
        axes[0].set_xlabel('Rating (1–10)'); axes[0].set_ylabel('Count')
        axes[0].set_title('All Ratings Histogram')
        axes[0].grid(axis='y', linestyle='--', alpha=0.4)

        # value counts bar
        vc = self.ratings['Book-Rating'].value_counts().sort_index()
        axes[1].bar(vc.index.astype(str), vc.values, color='#e74c3c', edgecolor='white')
        axes[1].set_xlabel('Rating'); axes[1].set_ylabel('Count')
        axes[1].set_title('Rating Frequency')
        axes[1].grid(axis='y', linestyle='--', alpha=0.4)

        plt.tight_layout(); plt.show()

    def _plot_top_authors(self):
        merged = self.ratings.merge(self.books[['ISBN','Book-Author']], on='ISBN', how='inner')
        top = merged.groupby('Book-Author').size().nlargest(15)

        fig, ax = plt.subplots(figsize=(11, 6))
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top)))
        bars = ax.barh(top.index[::-1], top.values[::-1], color=colors[::-1], edgecolor='white')
        for bar, val in zip(bars, top.values[::-1]):
            ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2,
                    str(val), va='center', fontsize=8)
        ax.set_xlabel('Number of Ratings')
        ax.set_title('👤 Top 15 Most Rated Authors', fontsize=13, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout(); plt.show()

    def _plot_publications_per_decade(self):
        yr = self.books['Year-Of-Publication'].dropna()
        yr = yr[(yr >= 1900) & (yr <= 2024)]
        decade = (yr // 10 * 10).astype(int)
        counts = decade.value_counts().sort_index()

        fig, ax = plt.subplots(figsize=(11, 5))
        ax.bar(counts.index.astype(str), counts.values,
               color='#2ecc71', edgecolor='white', width=0.7)
        ax.set_xlabel('Decade'); ax.set_ylabel('Books Published')
        ax.set_title('📅 Books Published per Decade', fontsize=13, fontweight='bold')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout(); plt.show()

    def _plot_top_publishers(self):
        top = self.books['Publisher'].value_counts().head(12)
        colors = plt.cm.Set3(np.linspace(0, 1, len(top)))

        fig, ax = plt.subplots(figsize=(9, 7))
        wedges, texts, autotexts = ax.pie(
            top.values, labels=top.index, autopct='%1.1f%%',
            colors=colors, startangle=140,
            wedgeprops=dict(edgecolor='white', linewidth=1.5))
        for t in autotexts: t.set_fontsize(7)
        ax.set_title('🏢 Top 12 Publishers', fontsize=13, fontweight='bold')
        plt.tight_layout(); plt.show()

    def _plot_user_activity(self):
        activity = self.ratings.groupby('User-ID').size()
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(activity[activity <= 50], bins=25, color='#9b59b6', edgecolor='white')
        ax.set_xlabel('Books Rated per User'); ax.set_ylabel('Number of Users')
        ax.set_title('👥 User Activity Distribution (≤50 ratings)', fontsize=13, fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  RECOMMENDER ENGINE
# ─────────────────────────────────────────────────────────────────────────────
class BookRecommenderSystem:
    def __init__(self, books, ratings, users):
        self.books   = books.copy()
        self.ratings = ratings.copy()
        self.users   = users.copy()

        # build avg_rating per book
        avg = ratings.groupby('ISBN')['Book-Rating'].agg(['mean','count']).reset_index()
        avg.columns = ['ISBN','avg_rating','rating_count']
        self.books = self.books.merge(avg, on='ISBN', how='left')
        self.books['avg_rating']   = self.books['avg_rating'].fillna(0).round(2)
        self.books['rating_count'] = self.books['rating_count'].fillna(0).astype(int)

        # popularity score (weighted)
        C = self.books['avg_rating'].mean()
        m = self.books['rating_count'].quantile(0.70)
        self.books['popularity'] = (
            (self.books['rating_count'] / (self.books['rating_count'] + m)) * self.books['avg_rating'] +
            (m / (self.books['rating_count'] + m)) * C
        ).round(3)

        print("\n🔧 Building TF-IDF content model...")
        self._build_content_model()
        print("✅ Recommender ready!\n")

    def _build_content_model(self):
        # use top-N books by rating count for content model (memory friendly)
        top_books = self.books.nlargest(500, 'rating_count').reset_index(drop=True)
        self._content_books = top_books.copy()

        corpus = (top_books['Book-Title'].fillna('') + ' ' +
                  top_books['Book-Author'].fillna('') + ' ' +
                  top_books['Publisher'].fillna(''))
        tfidf = TfidfVectorizer(max_features=3000, stop_words='english')
        mat   = tfidf.fit_transform(corpus)
        self._sim_matrix = cosine_similarity(mat)

    # ── content-based ─────────────────────────────────────────────────────────
    def content_recommendations(self, isbn, n=5):
        cb = self._content_books
        if isbn not in cb['ISBN'].values:
            return pd.DataFrame()
        idx    = cb[cb['ISBN'] == isbn].index[0]
        scores = list(enumerate(self._sim_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:n+1]
        return cb.iloc[[i for i, _ in scores]][
            ['ISBN','Book-Title','Book-Author','Year-Of-Publication','avg_rating','rating_count']
        ].reset_index(drop=True)

    # ── collaborative (item-based) ────────────────────────────────────────────
    def collaborative_recommendations(self, user_id, n=5):
        user_id = int(user_id) if str(user_id).isdigit() else user_id
        user_books = self.ratings[self.ratings['User-ID'] == user_id]['ISBN'].tolist()
        if not user_books:
            return self.popular_books(n)

        # users who rated same books
        similar_users = self.ratings[
            (self.ratings['ISBN'].isin(user_books)) &
            (self.ratings['User-ID'] != user_id)
        ]['User-ID'].value_counts().head(20).index

        # books those users liked (rating >= 7)
        recs = self.ratings[
            (self.ratings['User-ID'].isin(similar_users)) &
            (~self.ratings['ISBN'].isin(user_books)) &
            (self.ratings['Book-Rating'] >= 7)
        ]['ISBN'].value_counts().head(n).index

        return self.books[self.books['ISBN'].isin(recs)][
            ['ISBN','Book-Title','Book-Author','Year-Of-Publication','avg_rating','rating_count']
        ].reset_index(drop=True)

    # ── popular ───────────────────────────────────────────────────────────────
    def popular_books(self, n=10):
        return self.books.nlargest(n, 'popularity')[
            ['ISBN','Book-Title','Book-Author','Year-Of-Publication','avg_rating','rating_count','popularity']
        ].reset_index(drop=True)

    # ── search ────────────────────────────────────────────────────────────────
    def search(self, query, field='title'):
        q = query.lower()
        col_map = {'title': 'Book-Title', 'author': 'Book-Author', 'publisher': 'Publisher'}
        col = col_map.get(field, 'Book-Title')
        mask = self.books[col].str.lower().str.contains(q, na=False)
        return self.books[mask][
            ['ISBN','Book-Title','Book-Author','Year-Of-Publication','avg_rating','rating_count']
        ].head(10).reset_index(drop=True)

    # ── display ───────────────────────────────────────────────────────────────
    @staticmethod
    def show(df, title="Results"):
        print(f"\n🎯 {title}")
        print("="*70)
        if df is None or len(df) == 0:
            print("  No results found."); return
        for i, row in df.iterrows():
            yr  = int(row['Year-Of-Publication']) if pd.notna(row.get('Year-Of-Publication')) else 'N/A'
            avg = f"{row['avg_rating']:.1f}" if pd.notna(row.get('avg_rating')) else 'N/A'
            cnt = int(row['rating_count']) if pd.notna(row.get('rating_count')) else 0
            print(f"  {i+1:>2}. 📚 {str(row['Book-Title'])[:50]}")
            print(f"      👤 {row['Book-Author']}  |  📅 {yr}  |  ⭐ {avg}/10  |  🗳️  {cnt} ratings")
            print()

    # ══════════════════════════════════════════════════════════════════════════
    #  CHARTS
    # ══════════════════════════════════════════════════════════════════════════
    def plot_top_books(self, n=15):
        df = self.popular_books(n)
        fig, ax = plt.subplots(figsize=(12, 7))
        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, n))
        bars = ax.barh(df['Book-Title'].str[:35][::-1], df['popularity'][::-1],
                       color=colors, edgecolor='white')
        for bar, val in zip(bars, df['popularity'][::-1]):
            ax.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=8)
        ax.set_xlabel('Popularity Score (Weighted Rating)')
        ax.set_title(f'🏆 Top {n} Books by Popularity', fontsize=13, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.4)
        plt.tight_layout(); plt.show()

    def plot_similarity_heatmap(self, n=12):
        df = self._content_books.nlargest(n, 'rating_count').reset_index(drop=True)
        indices = df.index.tolist()
        sim = self._sim_matrix[np.ix_(indices, indices)]
        labels = [t[:18] for t in df['Book-Title'].tolist()]

        fig, ax = plt.subplots(figsize=(11, 9))
        im = ax.imshow(sim, cmap='YlOrRd', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, label='Cosine Similarity')
        ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=7)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{sim[i,j]:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if sim[i,j] > 0.5 else 'black')
        ax.set_title('🔥 TF-IDF Cosine Similarity Heatmap (Top Books)', fontsize=12, fontweight='bold')
        plt.tight_layout(); plt.show()

    def plot_rating_vs_count(self):
        df = self.books[(self.books['rating_count'] >= 3) & (self.books['avg_rating'] > 0)]
        fig, ax = plt.subplots(figsize=(10, 6))
        sc = ax.scatter(df['rating_count'], df['avg_rating'],
                        alpha=0.4, s=15, c=df['avg_rating'], cmap='RdYlGn', vmin=1, vmax=10)
        plt.colorbar(sc, ax=ax, label='Avg Rating')
        ax.set_xlabel('Number of Ratings'); ax.set_ylabel('Average Rating (1–10)')
        ax.set_title('📈 Avg Rating vs Rating Count', fontsize=13, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(linestyle='--', alpha=0.3)
        plt.tight_layout(); plt.show()

    def plot_year_trend(self):
        df = self.books[(self.books['Year-Of-Publication'] >= 1950) &
                        (self.books['Year-Of-Publication'] <= 2024) &
                        (self.books['avg_rating'] > 0)]
        yr_avg = df.groupby('Year-Of-Publication')['avg_rating'].mean()
        yr_cnt = df.groupby('Year-Of-Publication').size()

        fig, ax1 = plt.subplots(figsize=(12, 5))
        ax2 = ax1.twinx()
        ax1.plot(yr_avg.index, yr_avg.values, color='#e74c3c', linewidth=2, label='Avg Rating')
        ax2.bar(yr_cnt.index, yr_cnt.values, alpha=0.3, color='#3498db', label='Book Count')
        ax1.set_xlabel('Year'); ax1.set_ylabel('Average Rating', color='#e74c3c')
        ax2.set_ylabel('Books in Dataset', color='#3498db')
        ax1.set_title('📅 Publication Year Trend (Rating & Volume)', fontsize=13, fontweight='bold')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, loc='upper left')
        plt.tight_layout(); plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN MENU
# ─────────────────────────────────────────────────────────────────────────────
def main():
    print_banner()

    loader = DataLoader()
    books, ratings, users = loader.load()

    # ── EDA at startup ────────────────────────────────────────────────────────
    eda = EDA(books, ratings, users)
    eda.summary()

    print("\n📊 Show EDA charts now? (y/n): ", end='')
    if input().strip().lower() == 'y':
        eda.run_all_charts()

    rec = BookRecommenderSystem(books, ratings, users)

    while True:
        print("\n" + "="*62)
        print("  📚  BOOK RECOMMENDER  –  MAIN MENU")
        print("="*62)
        print("  1.  🔍  Search Books (title / author / publisher)")
        print("  2.  🎯  Content-Based Recommendations (by ISBN)")
        print("  3.  👥  Collaborative Recommendations (by User-ID)")
        print("  4.  🏆  Popular Books")
        print("─"*62)
        print("  5.  📊  [CHART] Top Books by Popularity")
        print("  6.  📊  [CHART] Similarity Heatmap")
        print("  7.  📊  [CHART] Avg Rating vs Rating Count")
        print("  8.  📊  [CHART] Year Trend (Rating & Volume)")
        print("  9.  📊  [CHART] Re-run EDA Charts")
        print("─"*62)
        print(" 10.  ❌  Exit")
        print("="*62)

        choice = input("\n🎯 Choice (1-10): ").strip()

        if choice == '1':
            q  = input("🔍 Query: ").strip()
            ft = input("Field (title/author/publisher) [title]: ").strip() or 'title'
            rec.show(rec.search(q, ft), f"Search: '{q}'")

        elif choice == '2':
            isbn = input("📚 Enter ISBN: ").strip()
            # show the book first
            bk = rec.books[rec.books['ISBN'] == isbn]
            if bk.empty:
                print("❌ ISBN not found. Try searching first (option 1).")
            else:
                row = bk.iloc[0]
                print(f"\n  📖 {row['Book-Title']}  by  {row['Book-Author']}")
                rec.show(rec.content_recommendations(isbn), "Content-Based Recommendations")

        elif choice == '3':
            uid = input("👤 User-ID: ").strip()
            rec.show(rec.collaborative_recommendations(uid), f"Recommendations for User {uid}")

        elif choice == '4':
            n = input("How many? [10]: ").strip()
            n = int(n) if n.isdigit() else 10
            rec.show(rec.popular_books(n), f"Top {n} Popular Books")

        elif choice == '5':
            rec.plot_top_books()

        elif choice == '6':
            rec.plot_similarity_heatmap()

        elif choice == '7':
            rec.plot_rating_vs_count()

        elif choice == '8':
            rec.plot_year_trend()

        elif choice == '9':
            eda.run_all_charts()

        elif choice == '10':
            print("\n📚 Goodbye!")
            break

        else:
            print("❌ Invalid choice.")

        input("\n⏸️  Press Enter to continue...")


if __name__ == '__main__':
    main()
