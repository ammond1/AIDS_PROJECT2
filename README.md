# AIDS_PROJECT2

🚀 **AIDS_PROJECT2** is a data-driven analysis project focused on AIDS-related datasets, combining data persistence, web scraping (Reddit), and AI-powered summarization for insights.

## 📂 Project Structure

```
AIDS_PROJECT2/
├── data/              # Data sources and outputs
├── others/
│   ├── persist/       # Persistent data storage (excluded from Git)
│   └── other files    # Additional resources
├── scripts/           # Project Python scripts
│   ├── reddit_scraper.py
│   └── summarize.py
├── requirements.txt   # Project dependencies
├── .gitignore         # Ignored files and folders
└── README.md          # Project overview
```

## ✨ Features

- 🔍 **Reddit Scraper**  
  Collects relevant Reddit posts using keyword-based search via PRAW.
- 🧠 **AI Summarization**  
  Uses GPT to analyze and summarize Reddit posts for quick insights.
- 🗄️ **Data Persistence**  
  Persistent storage structure to save and manage scraped data.
- 🌿 **Modular Scripts**  
  Clean and reusable Python scripts for scraping, summarizing, and analysis.

## 🛠️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/ammond1/AIDS_PROJECT2.git
cd AIDS_PROJECT2
```

2. **Create a virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # For Mac/Linux
venv\Scripts\activate     # For Windows
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables:**
Create a `.env` file in the project root and add your API keys:
```
REDDIT_ID=your_reddit_client_id
REDDIT_SECRET=your_reddit_client_secret
OPENAI_API_KEY=your_openai_api_key
```

## 🚀 Usage

Run the Reddit scraper and summarizer:
```bash
python scripts/reddit_scraper.py
```

The script will:
- Search Reddit posts with your query.
- Summarize posts using OpenAI.
- Print summaries and optionally save data for further analysis.

## 📦 Requirements

- Python 3.x
- PRAW
- OpenAI Python SDK
- python-dotenv

(See `requirements.txt` for full list.)

## 🤖 Future Improvements

- [ ] Add sentiment analysis of Reddit posts
- [ ] Visualize summarized data
- [ ] Automate regular scraping & reporting
- [ ] Add CSV/JSON export of summaries

## 🧩 Contributing

Contributions are welcome!  
Feel free to open issues or submit pull requests.

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙌 Acknowledgements

- [PRAW](https://praw.readthedocs.io/) — Python Reddit API Wrapper
- [OpenAI](https://platform.openai.com/docs) — GPT API
- [Python-dotenv](https://github.com/theskumar/python-dotenv) — Environment variable management

> *Data is scraped for educational and research purposes only. Please respect Reddit's API usage policies.*
