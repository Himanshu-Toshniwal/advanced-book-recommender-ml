# 📚 Advanced Book Recommender System with CI/CD Pipeline

A sophisticated machine learning-powered book recommendation system built with Python, featuring content-based filtering, collaborative filtering, and hybrid approaches with complete Docker containerization and CI/CD pipeline.

## 🚀 Features

- **🤖 Machine Learning Algorithms**: TF-IDF, Cosine Similarity, SVD Matrix Factorization
- **🎯 Multiple Recommendation Types**: Content-based, Collaborative filtering, Hybrid approach
- **📚 Comprehensive Book Database**: 20+ books across multiple genres
- **🔍 Advanced Search**: Multi-field search with fuzzy matching
- **📊 Reading Analytics**: Personal statistics and insights
- **🐳 Docker Support**: Complete containerization with multi-platform builds
- **🔄 CI/CD Pipeline**: Automated testing, security scanning, and deployment

## 📁 Project Structure

```
book-recommender-system/
├── .github/
│   └── workflows/
│       └── docker-build-deploy.yml # Docker CI/CD pipeline
├── src/
│   └── main.py                 # Book recommender system
├── tests/
│   └── test_main.py            # Comprehensive unit tests
├── Dockerfile                  # Docker container configuration
├── docker-compose.yml          # Multi-container setup
├── .dockerignore               # Docker ignore rules
├── requirements.txt            # Python ML dependencies
├── DOCKER_SETUP.md             # Docker deployment guide
└── README.md                   # Project documentation
```

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ci-cd-pipeline-getting-started
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Usage

### 🚀 Quick Start
```bash
# Local development
python src/main.py

# Docker (recommended)
docker run -p 8000:8000 your_username/book-recommender-system:latest

# Docker Compose
docker-compose up
```

### 📚 Menu Options
1. **🔍 Search Books** - Find books by title, author, or genre
2. **📚 Get Book Details** - View complete book information
3. **🎯 Content-Based Recommendations** - ML-powered suggestions
4. **👥 Collaborative Filtering** - User-based recommendations
5. **🔀 Hybrid Recommendations** - Best of both algorithms
6. **⭐ Rate a Book** - Add your ratings for better recommendations
7. **📊 Reading Statistics** - Personal analytics and insights
8. **🏆 Popular Books** - Trending and highly-rated books

### 📝 Example Usage
```
📚 ADVANCED BOOK RECOMMENDER SYSTEM
============================================================
1. 🔍 Search Books
2. 📚 Get Book Details
3. 🎯 Content-Based Recommendations
4. 👥 Collaborative Filtering Recommendations
5. 🔀 Hybrid Recommendations

🎯 Enter your choice (1-12): 3
📚 Enter Book ID for recommendations: 1

📚 Based on: To Kill a Mockingbird by Harper Lee
🎯 Content-Based Recommendations
============================================================
1. 📚 The Kite Runner by Khaled Hosseini
   🏷️ Contemporary Fiction | ⭐ 4.3/5.0 | 📅 2003
```

## 🧪 Testing

### Run Tests
```bash
pytest
```

### Run Tests with Coverage
```bash
pytest --cov=src tests/ --cov-report=html
```

### Test Coverage
The project includes comprehensive tests for:
- **ML Algorithm Tests** - TF-IDF, cosine similarity, SVD
- **Recommendation Engine** - Content-based, collaborative filtering
- **Search Functionality** - Multi-field search with fuzzy matching
- **User Rating System** - Rating validation and storage
- **Book Database** - CRUD operations and data integrity
- **Performance Tests** - Algorithm efficiency and scalability
- **Integration Tests** - End-to-end recommendation workflows

## 🔄 CI/CD Pipeline

The project uses GitHub Actions for automated Docker-based CI/CD:

### 🐳 Docker Pipeline Steps
1. **🧪 Code Testing** - Pytest with coverage reporting
2. **🔒 Security Scanning** - Trivy vulnerability scanner
3. **🐳 Docker Build** - Multi-platform image creation (AMD64 + ARM64)
4. **📦 DockerHub Push** - Automatic image publishing
5. **🚀 Production Deploy** - Container deployment
6. **🗑️ Cleanup** - Old image removal

### 🔄 Workflow Triggers
- Push to `main` or `develop` branch
- Pull requests to `main` branch
- Release creation
- Manual workflow dispatch

### 🔑 Required Secrets
```
DOCKERHUB_USERNAME = your_dockerhub_username
DOCKERHUB_TOKEN = your_dockerhub_access_token
```

## 📊 Data Storage

Books are stored in pandas DataFrame with comprehensive metadata:
```json
{
  "id": 1,
  "title": "To Kill a Mockingbird",
  "author": "Harper Lee",
  "genre": "Fiction",
  "year": 1960,
  "description": "A gripping tale of racial injustice and childhood innocence",
  "rating": 4.3,
  "pages": 376,
  "language": "English"
}
```

### 🤖 ML Models
- **TF-IDF Matrix**: Content vectorization for similarity calculation
- **Cosine Similarity**: Book-to-book similarity scores
- **User Ratings**: Collaborative filtering data
- **SVD Model**: Matrix factorization for dimensionality reduction

## 🏗️ Architecture

### 🤖 BookRecommenderSystem Class
- `__init__()` - Initialize ML models and book database
- `content_based_recommendations()` - TF-IDF + cosine similarity
- `collaborative_filtering_recommendations()` - User-based filtering
- `hybrid_recommendations()` - Combined approach
- `search_books()` - Multi-field fuzzy search
- `get_reading_statistics()` - User analytics
- `add_user_rating()` - Rating system

### 🔍 ML Components
- **TfidfVectorizer**: Content feature extraction
- **Cosine Similarity**: Book similarity calculation
- **TruncatedSVD**: Matrix factorization
- **Pandas DataFrame**: Efficient data operations

## 🔧 Development

### Prerequisites
- Python 3.10+
- Docker & Docker Compose
- pandas, numpy, scikit-learn
- pytest for testing

### 🐳 Docker Development
```bash
# Build and run locally
docker build -t book-recommender .
docker run -p 8000:8000 book-recommender

# Development with Docker Compose
docker-compose up --build
```

### Code Style
- Follows PEP 8 standards
- Linted with flake8
- Maximum line length: 127 characters
- Type hints for ML functions




---

**Built with ❤️ using Python, Machine Learning, Docker, and GitHub Actions**
