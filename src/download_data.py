"""
Data downloader for Book-Crossing dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset

Run this ONCE before main.py:
    python src/download_data.py
"""

import os
import sys
import zipfile

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
REQUIRED_FILES = ['BX-Books.csv', 'BX-Ratings.csv', 'BX-Users.csv']


def files_exist():
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED_FILES)


def download_via_kaggle():
    """Try downloading using kaggle CLI / API."""
    try:
        import kaggle  # noqa: F401
    except ImportError:
        print("⚠️  kaggle package not found. Installing...")
        os.system(f"{sys.executable} -m pip install kaggle -q")

    try:
        import kaggle
        print("📥 Downloading Book-Crossing dataset from Kaggle...")
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(
            'arashnic/book-recommendation-dataset',
            path=DATA_DIR,
            unzip=True,
            quiet=False
        )
        print("✅ Download complete!")
        return True
    except Exception as e:
        print(f"❌ Kaggle download failed: {e}")
        return False


def create_sample_data():
    """
    Create realistic sample CSVs that mimic Book-Crossing format.
    Used as fallback when Kaggle credentials are not available.
    """
    import pandas as pd
    import numpy as np

    print("📦 Creating sample Book-Crossing-format data...")

    books_data = [
        ('0195153448', 'Classical Mythology',          'Mark P. O. Morford',       2002, 'Oxford University Press',    'http://img.com/1.jpg'),
        ('0002005018', 'Clara Callan',                 'Richard Bruce Wright',     2001, 'HarperFlamingo Canada',      'http://img.com/2.jpg'),
        ('0060973129', 'Decision in Normandy',         'Carlo D\'Este',            1991, 'HarperPerennial',            'http://img.com/3.jpg'),
        ('0374157065', 'Flu: The Story of...',         'Gina Bari Kolata',         1999, 'Farrar Straus Giroux',       'http://img.com/4.jpg'),
        ('0393045218', 'The Mummies of Urumchi',       'E. J. W. Barber',          1999, 'W. W. Norton Company',       'http://img.com/5.jpg'),
        ('0399135782', 'The Kitchen God\'s Wife',      'Amy Tan',                  1991, 'Putnam Pub Group',           'http://img.com/6.jpg'),
        ('0425176428', 'What If?: The World\'s...',    'Robert Cowley',            2000, 'Berkley Publishing Group',   'http://img.com/7.jpg'),
        ('0671870432', 'PLEADING GUILTY',              'Scott Turow',              1993, 'Audioworks',                 'http://img.com/8.jpg'),
        ('0679425608', 'Under the Black Flag',         'David Cordingly',          1996, 'Random House',               'http://img.com/9.jpg'),
        ('074322678X', 'Where You\'ll Find Me',        'Ann Beattie',              2002, 'Scribner',                   'http://img.com/10.jpg'),
        ('0743206398', 'The Surgeon',                  'Tess Gerritsen',           2001, 'Pocket Books',               'http://img.com/11.jpg'),
        ('0743226860', 'Tis: A Memoir',                'Frank McCourt',            1999, 'Scribner',                   'http://img.com/12.jpg'),
        ('0743233441', 'The Lovely Bones',             'Alice Sebold',             2002, 'Little Brown',               'http://img.com/13.jpg'),
        ('0743247531', 'The Da Vinci Code',            'Dan Brown',                2003, 'Doubleday',                  'http://img.com/14.jpg'),
        ('0060928336', 'Divine Secrets of the Ya-Ya',  'Rebecca Wells',            1997, 'HarperCollins',              'http://img.com/15.jpg'),
        ('0316769177', 'The Catcher in the Rye',       'J.D. Salinger',            1951, 'Little Brown',               'http://img.com/16.jpg'),
        ('0679720200', 'Kafka on the Shore',           'Haruki Murakami',          2002, 'Vintage',                    'http://img.com/17.jpg'),
        ('0385504209', 'The Alchemist',                'Paulo Coelho',             1988, 'HarperOne',                  'http://img.com/18.jpg'),
        ('0060850523', 'Blink',                        'Malcolm Gladwell',         2005, 'Little Brown',               'http://img.com/19.jpg'),
        ('0316346624', 'The Tipping Point',            'Malcolm Gladwell',         2000, 'Little Brown',               'http://img.com/20.jpg'),
        ('0743273567', 'Angels and Demons',            'Dan Brown',                2000, 'Pocket Books',               'http://img.com/21.jpg'),
        ('0060931647', 'Harry Potter Sorcerer Stone',  'J.K. Rowling',             1997, 'Scholastic',                 'http://img.com/22.jpg'),
        ('043935806X', 'Harry Potter Chamber Secrets', 'J.K. Rowling',             1998, 'Scholastic',                 'http://img.com/23.jpg'),
        ('0439136369', 'Harry Potter Prisoner Azkaban','J.K. Rowling',             1999, 'Scholastic',                 'http://img.com/24.jpg'),
        ('0439139597', 'Harry Potter Goblet of Fire',  'J.K. Rowling',             2000, 'Scholastic',                 'http://img.com/25.jpg'),
        ('0439358078', 'Harry Potter Order Phoenix',   'J.K. Rowling',             2003, 'Scholastic',                 'http://img.com/26.jpg'),
        ('0060850529', 'Outliers',                     'Malcolm Gladwell',         2008, 'Little Brown',               'http://img.com/27.jpg'),
        ('0385333498', 'The Pillars of the Earth',     'Ken Follett',              1989, 'New American Library',       'http://img.com/28.jpg'),
        ('0312195516', 'The Girl with Dragon Tattoo',  'Stieg Larsson',            2005, 'Knopf',                      'http://img.com/29.jpg'),
        ('0307277674', '1Q84',                         'Haruki Murakami',          2009, 'Knopf',                      'http://img.com/30.jpg'),
        ('0385737951', 'The Hunger Games',             'Suzanne Collins',          2008, 'Scholastic',                 'http://img.com/31.jpg'),
        ('0439023483', 'The Hunger Games: Catching',   'Suzanne Collins',          2009, 'Scholastic',                 'http://img.com/32.jpg'),
        ('0439023513', 'Mockingjay',                   'Suzanne Collins',          2010, 'Scholastic',                 'http://img.com/33.jpg'),
        ('0307588378', 'Gone Girl',                    'Gillian Flynn',            2012, 'Crown',                      'http://img.com/34.jpg'),
        ('0316346629', 'Big Little Lies',              'Liane Moriarty',           2014, 'Flatiron Books',             'http://img.com/35.jpg'),
        ('0385490585', 'Dune',                         'Frank Herbert',            1965, 'Ace Books',                  'http://img.com/36.jpg'),
        ('0618640150', 'The Lord of the Rings',        'J.R.R. Tolkien',           1954, 'Houghton Mifflin',           'http://img.com/37.jpg'),
        ('0743298020', 'A Game of Thrones',            'George R.R. Martin',       1996, 'Bantam Books',               'http://img.com/38.jpg'),
        ('0553573403', 'A Clash of Kings',             'George R.R. Martin',       1998, 'Bantam Books',               'http://img.com/39.jpg'),
        ('0553106635', 'A Storm of Swords',            'George R.R. Martin',       2000, 'Bantam Books',               'http://img.com/40.jpg'),
        ('0385472579', 'Jurassic Park',                'Michael Crichton',         1990, 'Ballantine Books',           'http://img.com/41.jpg'),
        ('0679720201', 'Norwegian Wood',               'Haruki Murakami',          1987, 'Vintage',                    'http://img.com/42.jpg'),
        ('0316769487', 'Catch-22',                     'Joseph Heller',            1961, 'Simon Schuster',             'http://img.com/43.jpg'),
        ('0743273516', 'To Kill a Mockingbird',        'Harper Lee',               1960, 'Grand Central Publishing',   'http://img.com/44.jpg'),
        ('0451524934', '1984',                         'George Orwell',            1949, 'Signet Classic',             'http://img.com/45.jpg'),
        ('0743273513', 'Pride and Prejudice',          'Jane Austen',              1813, 'Modern Library',             'http://img.com/46.jpg'),
        ('0316769178', 'The Great Gatsby',             'F. Scott Fitzgerald',      1925, 'Scribner',                   'http://img.com/47.jpg'),
        ('0385333490', 'Sapiens',                      'Yuval Noah Harari',        2011, 'Harper',                     'http://img.com/48.jpg'),
        ('0735224293', 'Atomic Habits',                'James Clear',              2018, 'Avery',                      'http://img.com/49.jpg'),
        ('1501156700', 'It Ends with Us',              'Colleen Hoover',           2016, 'Atria Books',                'http://img.com/50.jpg'),
    ]

    books_df = pd.DataFrame(books_data, columns=[
        'ISBN', 'Book-Title', 'Book-Author', 'Year-Of-Publication',
        'Publisher', 'Image-URL-M'
    ])

    # Users
    np.random.seed(42)
    n_users = 200
    users_df = pd.DataFrame({
        'User-ID': range(1, n_users + 1),
        'Location': np.random.choice([
            'new york, usa', 'london, uk', 'toronto, canada',
            'sydney, australia', 'berlin, germany', 'paris, france',
            'mumbai, india', 'tokyo, japan', 'dubai, uae', 'singapore'
        ], n_users),
        'Age': np.random.choice(list(range(15, 70)) + [None]*20, n_users)
    })

    # Ratings
    isbns = books_df['ISBN'].tolist()
    ratings_rows = []
    for uid in range(1, n_users + 1):
        n_rated = np.random.randint(2, 15)
        rated_isbns = np.random.choice(isbns, n_rated, replace=False)
        for isbn in rated_isbns:
            rating = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                       p=[0.1, 0.02, 0.03, 0.05, 0.05, 0.1, 0.1, 0.15, 0.2, 0.1, 0.1])
            ratings_rows.append({'User-ID': uid, 'ISBN': isbn, 'Book-Rating': rating})

    ratings_df = pd.DataFrame(ratings_rows)

    os.makedirs(DATA_DIR, exist_ok=True)
    books_df.to_csv(os.path.join(DATA_DIR, 'BX-Books.csv'), index=False, sep=';')
    users_df.to_csv(os.path.join(DATA_DIR, 'BX-Users.csv'), index=False, sep=';')
    ratings_df.to_csv(os.path.join(DATA_DIR, 'BX-Ratings.csv'), index=False, sep=';')

    print(f"✅ Sample data created in {DATA_DIR}/")
    print(f"   📚 Books   : {len(books_df)}")
    print(f"   👤 Users   : {len(users_df)}")
    print(f"   ⭐ Ratings : {len(ratings_df)}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    if files_exist():
        print("✅ Dataset already present in data/ folder.")
        return

    print("="*55)
    print("  BOOK-CROSSING DATASET SETUP")
    print("="*55)
    print("\nOption 1: Auto-download from Kaggle (needs API key)")
    print("Option 2: Use built-in sample data (no setup needed)")
    print("\nDo you have a Kaggle API key configured? (y/n): ", end='', flush=True)
    try:
        ans = input().strip().lower()
    except EOFError:
        ans = 'n'  # non-interactive (Docker build / CI)

    if ans == 'y':
        success = download_via_kaggle()
        if not success:
            print("\n⚠️  Falling back to sample data...")
            create_sample_data()
    else:
        create_sample_data()
        print("\n💡 To use real Kaggle data later:")
        print("   1. Go to https://www.kaggle.com/datasets/arashnic/book-recommendation-dataset")
        print("   2. Download and extract CSVs into the data/ folder")
        print("   3. Re-run the app")


if __name__ == '__main__':
    main()
