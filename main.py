import shutil
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
from googleapiclient.discovery import build
import praw
from collections import Counter
import requests
from youtube_search import YoutubeSearch
import ast
from fuzzywuzzy import fuzz
from openai import OpenAI
import firebase_admin
from firebase_admin import credentials, db
load_dotenv()
# Initialize Firebase


cred = credentials.Certificate("/Users/admin/Desktop/AIDS 2/aids-project-2-25dcd-firebase-adminsdk-fbsvc-6788c98952.json")
firebase_admin.initialize_app(cred, {
     'databaseURL': 'https://aids-project-2-25dcd-default-rtdb.asia-southeast1.firebasedatabase.app'
 })
# Load environment variables


# Instantiate OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# --- PDF Analysis (currently not called in main flow) ---
def pdf_analysis_query(paths, model, prompt):
    loaders = [PyPDFLoader(path) for path in paths]
    documents = []
    for loader in loaders:
        documents.extend(loader.load())
    
    print(f"{len(loaders)} documents loaded with {len(documents)} pages.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
    splits = text_splitter.split_documents(documents)

    print("Number of splits:", len(splits))

    embedding = OpenAIEmbeddings()

    # Reset persistence directory
    try:
        shutil.rmtree('others/persist')
        print("Deleted previous store.")
    except:
        print("No previous store found.")

    persist_directory = './others/persist'

    vectordb = Chroma.from_documents(
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()

    print(f"Persist directory created. Size of Vector DB: {vectordb._collection.count()}")

    llm = ChatOpenAI(model_name=model, temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 4}),
        return_source_documents=True
    )

    result = qa_chain({"query": prompt})
    return result

# --- Reddit Scraper ---



def reddit_scrap_with_negatives(product_name, negative_keywords, number_of_posts=100, nickname_variants=None):
    """Scrape Reddit posts and comments for negative mentions of a product."""
    # print (nickname_variants)
    # Initialize Reddit API
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_ID'),
        client_secret=os.getenv('REDDIT_SECRET'),
        user_agent='ProductSentimentScraperV2'
    )

    # Prepare search variants
    nickname_variants = nickname_variants or []
    search_variants = [product_name.lower()] + [v.lower() for v in nickname_variants]

    # Find target subreddits
    target_subreddits = dynamic_subreddit_search(product_name)

    print(f"\n🎯 Target subreddits: {target_subreddits}\n")

    keyword_counter = {}

    # Search each subreddit
    for subreddit_name in target_subreddits:
        print(f"🔎 Searching subreddit: r/{subreddit_name}")
        subreddit = reddit.subreddit(subreddit_name)

        #search each variant
        for variant in search_variants:
            variant = str(variant)  # Ensure variant is a string
            print(f"🔍 Search term: {variant}")

            try:
                # Reddit search
                for submission in subreddit.search(variant, sort='new', limit = number_of_posts):
                    content = f"{submission.title or ''} {submission.selftext or ''}".lower()

                    # Check post content
                    for keyword in negative_keywords:
                        if keyword.lower() in content:
                            if keyword.lower() not in keyword_counter:
                                keyword_counter[keyword.lower()] = 0
                                keyword_counter[keyword.lower()] +=1
                            if keyword.lower() in content:
                                keyword_counter[keyword.lower()] += 1


                    # Check comments
                    try:
                        submission.comments.replace_more(limit=None)
                        for comment in submission.comments.list():
                            comment_body = comment.body.lower()
                            for keyword in negative_keywords:
                                if keyword.lower() in comment_body:
                                    if keyword.lower() not in keyword_counter:
                                        keyword_counter[keyword.lower()] = 0
                                        keyword_counter[keyword.lower()] +=1
                                    if keyword.lower() in content:
                                        keyword_counter[keyword.lower()] += 1
                    except Exception as e:
                        print(f"⚠️ Error loading comments: {e}")

            except Exception as e:
                print(f"⚠️ Error searching subreddit '{subreddit_name}' with variant '{variant}': {e}")

    # Final Results
    print(f"\n📊 Final Results for: {product_name}")
    print("-" * 40)
    for keyword, count in keyword_counter.items():
        print(f"{keyword}: {count} mentions")

    return keyword_counter


def dynamic_subreddit_search(query):
    """Find subreddits related to the product."""
    url = f"https://www.reddit.com/subreddits/search.json?q={query}&limit=8"
    headers = {"User-Agent": "Mozilla/5.0"}

    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        subreddits = []
        for child in data['data']['children']:
            subreddits.append(child['data']['display_name'])
        return subreddits
    except Exception as e:
        print(f"Dynamic subreddit search failed: {e}")
        return []
    
def generate_nicknames(product_name):
    prompt = f"""
    Give me a list of 5 common nickname variants or short forms that people online might use to refer to "{product_name}". 
    Include abbreviations, model names, brand shorthands, etc.
    Respond only as a valid Python list of strings.
    """
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    )
    nicknames = eval(response.choices[0].message.content)
    return nicknames

# --- YouTube Search ---
def search_youtube(product_name, max_results=10):
    results = YoutubeSearch(f"{product_name} review", max_results=max_results).to_dict()
    return results

# --- Fetch all YouTube comments (with nested replies) ---
def get_all_video_comments(video_id, api_key, max_comments=500):
    comments = []
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?key={api_key}&textFormat=plainText&part=snippet,replies&videoId={video_id}&maxResults=100"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for item in data.get('items', []):
            # Top-level comment
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
            # Replies to the comment
            if 'replies' in item:
                for reply in item['replies']['comments']:
                    comments.append(reply['snippet']['textDisplay'])
    else:
        print(f"Failed to fetch comments for {video_id}")
    
    return comments[:max_comments]

# --- Sentiment Analysis ---
NEGATIVE_WORDS = [
    "bad", "terrible", "horrible", "worst", "hate", "broken", "disappointed",
    "disappointing", "sucks", "awful", "problem", "issues", "trash", "scam", "ripoff", "annoying",
    "poor", "frustrating", "not good", "didn't work", "doesn't work", "hard to use",
    "waste of money", "misleading", "useless", "defective", "regret", "cheap quality",
    "returning it", "gave up", "not worth it"
]

def fast_sentiment(text):
    text_lower = text.lower()
    negative_hits = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
    return "negative" if negative_hits > 0 else "neutral"

def hybrid_sentiment(text, openai_client):
    fast_result = fast_sentiment(text)
    if fast_result == "negative":
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert."},
                {"role": "user", "content": f"Is the following comment expressing negativity towards a product? Reply only 'Yes' or 'No'.\n\nComment: {text}"}
            ]
        )
        reply = response.choices[0].message.content.strip().lower()
        return "negative" if "yes" in reply else "neutral"
    else:
        return "neutral"

# --- Comment Simplification ---
def simplify_comment(text):
    return text.lower().strip()

# --- Group similar issues ---
def group_similar_issues(issues, similarity_threshold=70):
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

# --- Analyze video (upgraded) ---
def analyze_video(video_id, api_key, openai_client):
    negatives = []
    comments = get_all_video_comments(video_id, api_key)

    for comment in comments:
        if hybrid_sentiment(comment, openai_client) == "negative":
            simplified = simplify_comment(comment)
            if simplified:
                negatives.append(simplified)

    final_negatives = group_similar_issues(negatives)
    return final_negatives

# --- Clean and extract final keywords ---
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

def handle_new_product(event):
    data = event.data
    print(event.path)
    if data:
        product_name = data.get('productName')
        if product_name:
            print(f"New product submitted: {product_name}")
            
            # Step 2: scrape Reddit
            results = main(product_name)
            
            # Step 3: push results back to Firebase
            product_ref = db.reference(f'products/{event.path}')  # use same key
            print(results)
            product_ref.update({
            'results': results
})
            print(f"Sent results back to Firebase.")

# Attach listener
products_ref = db.reference('products')
products_ref.listen(handle_new_product)

# --- MAIN ---
def main(product_name):
    YOUTUBE_API_KEY = os.getenv('GOOGLE_KEY')
    # product_name = input('Product Name:')
    videos = search_youtube(product_name)
    negatives = []

    for video in videos:
        title = video['title']
        video_id = video['id']
        print(f"\nAnalyzing video: {title}")

        negatives += analyze_video(video_id, YOUTUBE_API_KEY, client)

    print('Raw negatives collected:', negatives)

    print('cleaning comments')
    negatives = comment_clean(negatives, product_name)
    print('Cleaned keywords:', negatives)
    
    print('scraping reddit')
    return reddit_scrap_with_negatives(product_name, negatives, nickname_variants=generate_nicknames(product_name))

# if __name__ == "__main__":
#     main()