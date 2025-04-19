import shutil
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from googleapiclient.discovery import build
import praw
from openai import OpenAI
import re
from collections import Counter
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_search import YoutubeSearch
from textblob import TextBlob
from fuzzywuzzy import fuzz
import ast

#load .env file
load_dotenv()

#instantiate openai object
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

#not being used at the moment
def pdf_analysis_query(paths, model, prompt): 
    #model is to set what gpt model you want to use 
    #paths is an array of paths to documents

    loaders = []
    for path in paths:
        load = PyPDFLoader(path)
        loaders.append(load)

    document = []
    for loader in loaders:
        document.extend(loader.load())

    print(len(loaders), "documents loaded with", len(document), "pages in total.")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300
    )
    splits = text_splitter.split_documents(document)

    print("Number of splits in document loaded:", len(splits))

    embedding = OpenAIEmbeddings() #creates embedding model

    #check if a persist directory exists if yes delete to start a fresh one 
    try:
        shutil.rmtree('others/persist')       # remove old version, else can't work
        print("Deleting previous store")
    except:
        print("No store found")

    persist_directory = './others/persist'


    vectordb = Chroma.from_documents(
        documents=splits,                           # target the splits created from the documents loaded
        embedding=embedding,                        # use the OpenAI embedding specified
        persist_directory=persist_directory         # store in the persist directory for future use
    )

    vectordb.persist()                              # store vectordb

    print("Persist Directory created.")
    print("Size of Vector Database:", vectordb._collection.count())    # same as the number of splits

    llm = llm = ChatOpenAI(model_name=model, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
    llm,
    # retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 6}),
    retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
    return_source_documents=True
    )

    result = qa_chain({"query": prompt})
    return result


#reddit scrapper
def reddit_scrap_with_negatives(product_name, negative_keywords, number_of_posts=10000):

    # Set up Reddit API
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_ID'),
        client_secret=os.getenv('REDDIT_SECRET'),
        user_agent='Reddit scraper'
    )

    # Search Reddit
    subreddit = reddit.subreddit('all')
    posts = []

    for submission in subreddit.search(product_name, limit=number_of_posts):
        posts.append(submission.title + " " + (submission.selftext or ""))
    print( posts)
    # Analyze posts
    keyword_counter = Counter()

    for post in posts:
        post_lower = post.lower()
        for keyword in negative_keywords:
            # Use word boundaries to match full words
            if re.search(re.escape(keyword), post_lower):
                keyword_counter[keyword] += 1

    # Display results
    print(f"Product: {product_name}\n")
    print("Negative keyword mentions:")
    for keyword in negative_keywords:
        print(f"- {keyword}: {keyword_counter[keyword]} mentions")

def search_youtube(product_name, max_results=5):
    """Search YouTube for review videos."""
    results = YoutubeSearch(f"{product_name} review", max_results=max_results).to_dict()
    return results

def get_video_comments(video_id, api_key, max_comments=20):
    """Get comments from a YouTube video using YouTube Data API."""
    comments = []
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&textFormat=plainText&part=snippet&videoId={video_id}&maxResults={max_comments}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for item in data.get('items', []):
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
    else:
        print(f"Failed to fetch comments for {video_id}")
    
    return comments

def analyze_sentiment(text):
    """Simple sentiment analyzer: returns polarity."""
    blob = TextBlob(text)
    return blob.sentiment.polarity  # -1 (very negative) to +1 (very positive)

def simplify_comment(text):
    """Simplify a comment: lower, remove fluff, trim."""
    return text.lower().strip()

def group_similar_issues(issues, similarity_threshold=80):
    """Group very similar issues together based on fuzzy text matching."""
    grouped = []

    for issue in issues:
        found_similar = False
        for existing in grouped:
            if fuzz.token_set_ratio(issue, existing) >= similarity_threshold:
                found_similar = True
                break
        if not found_similar:
            grouped.append(issue)
    return grouped

def analyze_transcript(video_id):
    """Analyze the transcript to extract specific negative parts."""
    negatives = []

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        for entry in transcript:
            text = entry['text']
            if analyze_sentiment(text) < -0.1:
                negatives.append(text)
    except Exception as e:
        print(f"No transcript available for {video_id} ({e})")
    
    return negatives

def analyze_video(video_id, api_key):
    """Analyze transcript + comments for negativity."""
    negatives = []

    # # Analyze transcript
    # transcript_negatives = analyze_transcript(video_id)
    # for t in transcript_negatives:
    #     simplified = simplify_comment(t)
    #     if simplified:
    #         negatives.append(simplified)

    # Analyze comments
    comments = get_video_comments(video_id, api_key)
    for comment in comments:
        if analyze_sentiment(comment) < -0.1:
            simplified = simplify_comment(comment)
            if simplified:
                negatives.append(simplified)

    # ✨ Group very similar complaints
    final_negatives = group_similar_issues(negatives)

    return final_negatives

def comment_clean(negatives, product_name):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a professional data cleaner and keyword extractor."},
            {"role": "user", "content": f"""Given the following array:
{negatives}

For each entry:
- Check if the content is related to "{product_name}".
- If it is, extract the core meaning into 1-2 keywords (no sentences).
- If it is NOT related to "{product_name}", skip it (do not include it).

IMPORTANT: Only respond with a **valid Python array** (e.g., ['keyword1', 'keyword2', ...]) — no explanations, no extra text. Only the array."""}
        ]
    )
    return ast.literal_eval(response.choices[0].message.content)

def main():
    # Replace this with your YouTube API key
    YOUTUBE_API_KEY = os.getenv('GOOGLE_KEY')

    product_name = input("Enter product name: ")
    videos = search_youtube(product_name)
    negatives = []
    for video in videos:
        title = video['title']
        video_id = video['id']
        print(f"\nAnalyzing video: {title}")

        negatives += analyze_video(video_id, YOUTUBE_API_KEY)
        
    print(' look here',negatives)
    negatives = comment_clean(negatives, product_name)
    print(negatives)
    reddit_scrap_with_negatives(product_name, negatives)

if __name__ == "__main__":
    main()
