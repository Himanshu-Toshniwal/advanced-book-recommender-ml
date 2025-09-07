import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import sqlite3
import json
import random
from datetime import datetime
import re

class BookRecommenderSystem:
    def __init__(self):
        self.books_db = self.create_sample_database()
        self.user_ratings = {}
        self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        self.content_matrix = None
        self.similarity_matrix = None
        self.svd_model = TruncatedSVD(n_components=50)
        
        print("📚 Advanced Book Recommender System Initialized!")
        self.setup_recommendation_models()
    
    def create_sample_database(self):
        """Create comprehensive book database"""
        books = [
            # Fiction
            {'id': 1, 'title': 'To Kill a Mockingbird', 'author': 'Harper Lee', 'genre': 'Fiction', 'year': 1960, 
             'description': 'A gripping tale of racial injustice and childhood innocence in the American South',
             'rating': 4.3, 'pages': 376, 'language': 'English'},
            
            {'id': 2, 'title': '1984', 'author': 'George Orwell', 'genre': 'Dystopian Fiction', 'year': 1949,
             'description': 'A dystopian social science fiction novel about totalitarian control and surveillance',
             'rating': 4.2, 'pages': 328, 'language': 'English'},
            
            {'id': 3, 'title': 'Pride and Prejudice', 'author': 'Jane Austen', 'genre': 'Romance', 'year': 1813,
             'description': 'A romantic novel about manners, upbringing, morality, and marriage in society',
             'rating': 4.1, 'pages': 432, 'language': 'English'},
            
            # Science Fiction
            {'id': 4, 'title': 'Dune', 'author': 'Frank Herbert', 'genre': 'Science Fiction', 'year': 1965,
             'description': 'Epic science fiction novel set in the distant future amidst a feudal interstellar society',
             'rating': 4.4, 'pages': 688, 'language': 'English'},
            
            {'id': 5, 'title': 'The Hitchhiker\'s Guide to the Galaxy', 'author': 'Douglas Adams', 'genre': 'Science Fiction', 'year': 1979,
             'description': 'A comedic science fiction series about space travel and alien civilizations',
             'rating': 4.2, 'pages': 224, 'language': 'English'},
            
            # Fantasy
            {'id': 6, 'title': 'The Lord of the Rings', 'author': 'J.R.R. Tolkien', 'genre': 'Fantasy', 'year': 1954,
             'description': 'Epic high fantasy novel about the quest to destroy the One Ring and defeat Dark Lord',
             'rating': 4.5, 'pages': 1216, 'language': 'English'},
            
            {'id': 7, 'title': 'Harry Potter and the Philosopher\'s Stone', 'author': 'J.K. Rowling', 'genre': 'Fantasy', 'year': 1997,
             'description': 'Young wizard discovers his magical heritage and attends Hogwarts School of Witchcraft',
             'rating': 4.4, 'pages': 309, 'language': 'English'},
            
            # Mystery/Thriller
            {'id': 8, 'title': 'The Girl with the Dragon Tattoo', 'author': 'Stieg Larsson', 'genre': 'Mystery', 'year': 2005,
             'description': 'Psychological thriller about journalist and hacker investigating disappearance',
             'rating': 4.1, 'pages': 672, 'language': 'English'},
            
            {'id': 9, 'title': 'Gone Girl', 'author': 'Gillian Flynn', 'genre': 'Thriller', 'year': 2012,
             'description': 'Psychological thriller about marriage and media manipulation during missing person case',
             'rating': 4.0, 'pages': 432, 'language': 'English'},
            
            # Non-Fiction
            {'id': 10, 'title': 'Sapiens', 'author': 'Yuval Noah Harari', 'genre': 'Non-Fiction', 'year': 2011,
             'description': 'Brief history of humankind from Stone Age to present day',
             'rating': 4.3, 'pages': 512, 'language': 'English'},
            
            {'id': 11, 'title': 'Educated', 'author': 'Tara Westover', 'genre': 'Memoir', 'year': 2018,
             'description': 'Memoir about education, family, and the struggle between loyalty and independence',
             'rating': 4.2, 'pages': 334, 'language': 'English'},
            
            # Classic Literature
            {'id': 12, 'title': 'The Great Gatsby', 'author': 'F. Scott Fitzgerald', 'genre': 'Classic', 'year': 1925,
             'description': 'Story of mysterious millionaire Jay Gatsby and his obsession with Daisy Buchanan',
             'rating': 3.9, 'pages': 180, 'language': 'English'},
            
            {'id': 13, 'title': 'One Hundred Years of Solitude', 'author': 'Gabriel García Márquez', 'genre': 'Magical Realism', 'year': 1967,
             'description': 'Multi-generational story of Buendía family in fictional town of Macondo',
             'rating': 4.1, 'pages': 417, 'language': 'English'},
            
            # Contemporary Fiction
            {'id': 14, 'title': 'The Kite Runner', 'author': 'Khaled Hosseini', 'genre': 'Contemporary Fiction', 'year': 2003,
             'description': 'Story of friendship, guilt, and redemption set against backdrop of Afghanistan',
             'rating': 4.3, 'pages': 371, 'language': 'English'},
            
            {'id': 15, 'title': 'Life of Pi', 'author': 'Yann Martel', 'genre': 'Adventure', 'year': 2001,
             'description': 'Survival story of Indian boy stranded on lifeboat with Bengal tiger',
             'rating': 3.8, 'pages': 319, 'language': 'English'},
            
            # Horror
            {'id': 16, 'title': 'The Shining', 'author': 'Stephen King', 'genre': 'Horror', 'year': 1977,
             'description': 'Psychological horror about family isolated in haunted hotel during winter',
             'rating': 4.2, 'pages': 447, 'language': 'English'},
            
            # Biography
            {'id': 17, 'title': 'Steve Jobs', 'author': 'Walter Isaacson', 'genre': 'Biography', 'year': 2011,
             'description': 'Comprehensive biography of Apple co-founder and technology visionary',
             'rating': 4.1, 'pages': 656, 'language': 'English'},
            
            # Self-Help
            {'id': 18, 'title': 'Atomic Habits', 'author': 'James Clear', 'genre': 'Self-Help', 'year': 2018,
             'description': 'Guide to building good habits and breaking bad ones through small changes',
             'rating': 4.4, 'pages': 320, 'language': 'English'},
            
            # Historical Fiction
            {'id': 19, 'title': 'The Book Thief', 'author': 'Markus Zusak', 'genre': 'Historical Fiction', 'year': 2005,
             'description': 'Story narrated by Death about young girl living with foster family in Nazi Germany',
             'rating': 4.4, 'pages': 552, 'language': 'English'},
            
            {'id': 20, 'title': 'All Quiet on the Western Front', 'author': 'Erich Maria Remarque', 'genre': 'War Fiction', 'year': 1929,
             'description': 'Anti-war novel about German soldiers during World War I',
             'rating': 4.0, 'pages': 295, 'language': 'English'}
        ]
        
        return pd.DataFrame(books)
    
    def setup_recommendation_models(self):
        """Setup machine learning models for recommendations"""
        # Content-based filtering using TF-IDF
        combined_features = self.books_db['description'] + ' ' + self.books_db['genre'] + ' ' + self.books_db['author']
        self.content_matrix = self.tfidf_vectorizer.fit_transform(combined_features)
        self.similarity_matrix = cosine_similarity(self.content_matrix)
        
        print("✅ Recommendation models initialized!")
    
    def add_user_rating(self, user_id, book_id, rating):
        """Add user rating for collaborative filtering"""
        if user_id not in self.user_ratings:
            self.user_ratings[user_id] = {}
        
        self.user_ratings[user_id][book_id] = rating
        print(f"✅ Rating added: User {user_id} rated Book {book_id} with {rating} stars")
    
    def content_based_recommendations(self, book_id, num_recommendations=5):
        """Content-based recommendations using book features"""
        if book_id not in self.books_db['id'].values:
            return []
        
        book_idx = self.books_db[self.books_db['id'] == book_id].index[0]
        similarity_scores = list(enumerate(self.similarity_matrix[book_idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        
        # Get top similar books (excluding the book itself)
        similar_books = similarity_scores[1:num_recommendations+1]
        recommended_ids = [self.books_db.iloc[i[0]]['id'] for i in similar_books]
        
        return self.get_books_by_ids(recommended_ids)
    
    def collaborative_filtering_recommendations(self, user_id, num_recommendations=5):
        """Collaborative filtering based on user ratings"""
        if user_id not in self.user_ratings:
            return self.get_popular_books(num_recommendations)
        
        user_ratings = self.user_ratings[user_id]
        recommendations = []
        
        # Find similar users
        similar_users = self.find_similar_users(user_id)
        
        # Get books rated highly by similar users
        for similar_user in similar_users:
            for book_id, rating in self.user_ratings[similar_user].items():
                if book_id not in user_ratings and rating >= 4.0:
                    if book_id not in [r['id'] for r in recommendations]:
                        book_info = self.get_book_by_id(book_id)
                        if book_info:
                            recommendations.append(book_info)
        
        return recommendations[:num_recommendations]
    
    def find_similar_users(self, user_id, threshold=0.3):
        """Find users with similar reading preferences"""
        if user_id not in self.user_ratings:
            return []
        
        user_ratings = self.user_ratings[user_id]
        similar_users = []
        
        for other_user, other_ratings in self.user_ratings.items():
            if other_user == user_id:
                continue
            
            # Find common books
            common_books = set(user_ratings.keys()) & set(other_ratings.keys())
            
            if len(common_books) >= 2:  # Need at least 2 common books
                # Calculate similarity
                similarity = self.calculate_user_similarity(user_ratings, other_ratings, common_books)
                if similarity > threshold:
                    similar_users.append(other_user)
        
        return similar_users
    
    def calculate_user_similarity(self, ratings1, ratings2, common_books):
        """Calculate similarity between two users"""
        if not common_books:
            return 0
        
        sum1 = sum([ratings1[book] for book in common_books])
        sum2 = sum([ratings2[book] for book in common_books])
        
        sum1_sq = sum([ratings1[book]**2 for book in common_books])
        sum2_sq = sum([ratings2[book]**2 for book in common_books])
        
        sum_products = sum([ratings1[book] * ratings2[book] for book in common_books])
        
        n = len(common_books)
        numerator = sum_products - (sum1 * sum2 / n)
        denominator = ((sum1_sq - sum1**2/n) * (sum2_sq - sum2**2/n))**0.5
        
        if denominator == 0:
            return 0
        
        return numerator / denominator
    
    def hybrid_recommendations(self, user_id=None, book_id=None, num_recommendations=5):
        """Hybrid approach combining content-based and collaborative filtering"""
        recommendations = []
        
        # Get content-based recommendations
        if book_id:
            content_recs = self.content_based_recommendations(book_id, num_recommendations//2)
            recommendations.extend(content_recs)
        
        # Get collaborative filtering recommendations
        if user_id:
            collab_recs = self.collaborative_filtering_recommendations(user_id, num_recommendations//2)
            recommendations.extend(collab_recs)
        
        # Remove duplicates and limit results
        seen_ids = set()
        unique_recs = []
        for book in recommendations:
            if book['id'] not in seen_ids:
                unique_recs.append(book)
                seen_ids.add(book['id'])
        
        # Fill remaining slots with popular books if needed
        if len(unique_recs) < num_recommendations:
            popular_books = self.get_popular_books(num_recommendations - len(unique_recs))
            for book in popular_books:
                if book['id'] not in seen_ids:
                    unique_recs.append(book)
        
        return unique_recs[:num_recommendations]
    
    def search_books(self, query, search_type='all'):
        """Search books by title, author, or genre"""
        query = query.lower()
        results = []
        
        for _, book in self.books_db.iterrows():
            match = False
            
            if search_type in ['all', 'title'] and query in book['title'].lower():
                match = True
            elif search_type in ['all', 'author'] and query in book['author'].lower():
                match = True
            elif search_type in ['all', 'genre'] and query in book['genre'].lower():
                match = True
            elif search_type == 'description' and query in book['description'].lower():
                match = True
            
            if match:
                results.append(book.to_dict())
        
        return results
    
    def get_books_by_genre(self, genre, limit=10):
        """Get books by specific genre"""
        genre_books = self.books_db[self.books_db['genre'].str.contains(genre, case=False, na=False)]
        return genre_books.head(limit).to_dict('records')
    
    def get_books_by_author(self, author, limit=10):
        """Get books by specific author"""
        author_books = self.books_db[self.books_db['author'].str.contains(author, case=False, na=False)]
        return author_books.head(limit).to_dict('records')
    
    def get_popular_books(self, limit=5):
        """Get most popular books based on ratings"""
        popular = self.books_db.nlargest(limit, 'rating')
        return popular.to_dict('records')
    
    def get_recent_books(self, limit=5):
        """Get most recent books"""
        recent = self.books_db.nlargest(limit, 'year')
        return recent.to_dict('records')
    
    def get_book_by_id(self, book_id):
        """Get book details by ID"""
        book = self.books_db[self.books_db['id'] == book_id]
        if not book.empty:
            return book.iloc[0].to_dict()
        return None
    
    def get_books_by_ids(self, book_ids):
        """Get multiple books by their IDs"""
        books = self.books_db[self.books_db['id'].isin(book_ids)]
        return books.to_dict('records')
    
    def get_reading_statistics(self, user_id):
        """Get user's reading statistics"""
        if user_id not in self.user_ratings:
            return {"message": "No ratings found for this user"}
        
        ratings = self.user_ratings[user_id]
        
        # Get rated books details
        rated_book_ids = list(ratings.keys())
        rated_books = self.get_books_by_ids(rated_book_ids)
        
        # Calculate statistics
        total_books = len(ratings)
        avg_rating = sum(ratings.values()) / total_books if total_books > 0 else 0
        
        # Genre preferences
        genre_counts = {}
        total_pages = 0
        
        for book in rated_books:
            genre = book['genre']
            genre_counts[genre] = genre_counts.get(genre, 0) + 1
            total_pages += book['pages']
        
        favorite_genre = max(genre_counts, key=genre_counts.get) if genre_counts else "None"
        
        return {
            'total_books_rated': total_books,
            'average_rating': round(avg_rating, 2),
            'total_pages_read': total_pages,
            'favorite_genre': favorite_genre,
            'genre_distribution': genre_counts,
            'highest_rated_books': [book_id for book_id, rating in ratings.items() if rating == 5.0]
        }
    
    def get_book_recommendations_explanation(self, book_id, recommended_books):
        """Explain why books were recommended"""
        base_book = self.get_book_by_id(book_id)
        if not base_book:
            return []
        
        explanations = []
        for rec_book in recommended_books:
            reasons = []
            
            # Same genre
            if base_book['genre'] == rec_book['genre']:
                reasons.append(f"Same genre ({base_book['genre']})")
            
            # Same author
            if base_book['author'] == rec_book['author']:
                reasons.append(f"Same author ({base_book['author']})")
            
            # Similar rating
            if abs(base_book['rating'] - rec_book['rating']) <= 0.3:
                reasons.append("Similar rating")
            
            # Similar time period
            if abs(base_book['year'] - rec_book['year']) <= 20:
                reasons.append("Similar time period")
            
            explanations.append({
                'book': rec_book,
                'reasons': reasons if reasons else ["Content similarity"]
            })
        
        return explanations
    
    def display_book_details(self, book):
        """Display formatted book details"""
        print(f"\n📖 {book['title']}")
        print(f"👤 Author: {book['author']}")
        print(f"🏷️  Genre: {book['genre']}")
        print(f"📅 Year: {book['year']}")
        print(f"⭐ Rating: {book['rating']}/5.0")
        print(f"📄 Pages: {book['pages']}")
        print(f"📝 Description: {book['description']}")
        print("-" * 60)
    
    def display_recommendations(self, recommendations, title="Recommendations"):
        """Display formatted recommendations"""
        print(f"\n🎯 {title}")
        print("=" * 60)
        
        if not recommendations:
            print("No recommendations found.")
            return
        
        for i, book in enumerate(recommendations, 1):
            print(f"{i}. 📚 {book['title']} by {book['author']}")
            print(f"   🏷️  {book['genre']} | ⭐ {book['rating']}/5.0 | 📅 {book['year']}")
            print(f"   📝 {book['description'][:100]}...")
            print()

def main():
    """Main function to run the book recommender system"""
    recommender = BookRecommenderSystem()
    
    # Add some sample user ratings for demonstration
    sample_ratings = {
        'user1': {1: 5.0, 2: 4.0, 6: 5.0, 7: 4.5, 10: 4.0},
        'user2': {2: 5.0, 4: 4.5, 5: 4.0, 8: 3.5, 16: 4.0},
        'user3': {3: 4.0, 6: 5.0, 7: 5.0, 12: 3.5, 19: 4.5}
    }
    
    for user_id, ratings in sample_ratings.items():
        for book_id, rating in ratings.items():
            recommender.add_user_rating(user_id, book_id, rating)
    
    while True:
        print("\n" + "="*60)
        print("📚 ADVANCED BOOK RECOMMENDER SYSTEM")
        print("="*60)
        print("1. 🔍 Search Books")
        print("2. 📖 Get Book Details")
        print("3. 🎯 Content-Based Recommendations")
        print("4. 👥 Collaborative Filtering Recommendations")
        print("5. 🔀 Hybrid Recommendations")
        print("6. ⭐ Rate a Book")
        print("7. 📊 View Reading Statistics")
        print("8. 🏆 Popular Books")
        print("9. 🆕 Recent Books")
        print("10. 🏷️ Browse by Genre")
        print("11. 👤 Browse by Author")
        print("12. ❌ Exit")
        print("="*60)
        
        choice = input("\n🎯 Enter your choice (1-12): ").strip()
        
        if choice == '1':
            query = input("\n🔍 Enter search query: ").strip()
            search_type = input("Search by (title/author/genre/all): ").strip().lower()
            if search_type not in ['title', 'author', 'genre', 'all']:
                search_type = 'all'
            
            results = recommender.search_books(query, search_type)
            recommender.display_recommendations(results, f"Search Results for '{query}'")
        
        elif choice == '2':
            try:
                book_id = int(input("\n📖 Enter Book ID: "))
                book = recommender.get_book_by_id(book_id)
                if book:
                    recommender.display_book_details(book)
                else:
                    print("❌ Book not found!")
            except ValueError:
                print("❌ Please enter a valid book ID")
        
        elif choice == '3':
            try:
                book_id = int(input("\n📚 Enter Book ID for recommendations: "))
                recommendations = recommender.content_based_recommendations(book_id)
                
                if recommendations:
                    base_book = recommender.get_book_by_id(book_id)
                    print(f"\n📖 Based on: {base_book['title']} by {base_book['author']}")
                    recommender.display_recommendations(recommendations, "Content-Based Recommendations")
                    
                    # Show explanations
                    explanations = recommender.get_book_recommendations_explanation(book_id, recommendations)
                    print("\n💡 Why these books were recommended:")
                    for exp in explanations:
                        print(f"• {exp['book']['title']}: {', '.join(exp['reasons'])}")
                else:
                    print("❌ No recommendations found!")
            except ValueError:
                print("❌ Please enter a valid book ID")
        
        elif choice == '4':
            user_id = input("\n👤 Enter User ID: ").strip()
            recommendations = recommender.collaborative_filtering_recommendations(user_id)
            recommender.display_recommendations(recommendations, f"Recommendations for User {user_id}")
        
        elif choice == '5':
            user_id = input("\n👤 Enter User ID (optional): ").strip()
            book_id_input = input("📚 Enter Book ID (optional): ").strip()
            
            book_id = None
            if book_id_input:
                try:
                    book_id = int(book_id_input)
                except ValueError:
                    print("❌ Invalid book ID, ignoring...")
            
            user_id = user_id if user_id else None
            recommendations = recommender.hybrid_recommendations(user_id, book_id)
            recommender.display_recommendations(recommendations, "Hybrid Recommendations")
        
        elif choice == '6':
            user_id = input("\n👤 Enter User ID: ").strip()
            try:
                book_id = int(input("📚 Enter Book ID: "))
                rating = float(input("⭐ Enter Rating (1.0-5.0): "))
                
                if 1.0 <= rating <= 5.0:
                    recommender.add_user_rating(user_id, book_id, rating)
                else:
                    print("❌ Rating must be between 1.0 and 5.0")
            except ValueError:
                print("❌ Please enter valid book ID and rating")
        
        elif choice == '7':
            user_id = input("\n👤 Enter User ID: ").strip()
            stats = recommender.get_reading_statistics(user_id)
            
            print(f"\n📊 Reading Statistics for User {user_id}")
            print("="*50)
            
            if "message" in stats:
                print(stats["message"])
            else:
                print(f"📚 Total Books Rated: {stats['total_books_rated']}")
                print(f"⭐ Average Rating: {stats['average_rating']}")
                print(f"📄 Total Pages Read: {stats['total_pages_read']}")
                print(f"🏷️ Favorite Genre: {stats['favorite_genre']}")
                print(f"🏆 5-Star Books: {len(stats['highest_rated_books'])}")
                
                print("\n📊 Genre Distribution:")
                for genre, count in stats['genre_distribution'].items():
                    print(f"  • {genre}: {count} books")
        
        elif choice == '8':
            popular_books = recommender.get_popular_books(10)
            recommender.display_recommendations(popular_books, "Most Popular Books")
        
        elif choice == '9':
            recent_books = recommender.get_recent_books(10)
            recommender.display_recommendations(recent_books, "Recent Books")
        
        elif choice == '10':
            genre = input("\n🏷️ Enter Genre: ").strip()
            genre_books = recommender.get_books_by_genre(genre)
            recommender.display_recommendations(genre_books, f"Books in {genre}")
        
        elif choice == '11':
            author = input("\n👤 Enter Author Name: ").strip()
            author_books = recommender.get_books_by_author(author)
            recommender.display_recommendations(author_books, f"Books by {author}")
        
        elif choice == '12':
            print("\n📚 Thank you for using the Book Recommender System!")
            break
        
        else:
            print("❌ Invalid choice. Please try again.")
        
        input("\n⏸️ Press Enter to continue...")

if __name__ == "__main__":
    main()