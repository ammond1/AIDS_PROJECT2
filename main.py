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

load_dotenv()
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

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

def reddit_scrap(query,number_of_post):
    # Set up Reddit API
    reddit = praw.Reddit(
        client_id=os.getenv('REDDIT_ID'),
        client_secret=os.getenv('REDDIT_SECRET'),
        user_agent='Reddit scraper'
    )

    # Search Reddit
    subreddit = reddit.subreddit('all')  # or specific subreddit
    limit = number_of_post  # number of results you want

    posts = []
    for submission in subreddit.search(query, limit=limit):
        posts.append({
            'title': submission.title,
            'selftext': submission.selftext.strip() or "No text provided."
        })

    print(summarize_posts(posts,query))

def format_posts(posts):
    formatted = ""
    for idx, post in enumerate(posts, start=1):
        formatted += f"{idx}. Title: {post['title']}\n   Content: {post['selftext']}\n\n"
    return formatted

def summarize_posts(posts, item):
    formatted_posts = format_posts(posts)

    response = client.chat.completions.create(model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant who summarizes Reddit posts."},
        {"role": "user", "content": f"Summarize the following Reddit posts:\n\n{formatted_posts}, specifically focusing on what users think about {item}"}
    ])

    summary = response.choices[0].message.content
    return summary

# #set up for pdf_analysis_query
# path = []
# path.append('CHANGE TO YOUR DOC PATH')
# model = 'gpt-4o'
# prompt = "INPUT YOUR PROMPT"
# print(pdf_analysis_query(path, model,prompt))

#set up for reddit scrapper
# reddit_scrap('Logitech mx master 3', 5)