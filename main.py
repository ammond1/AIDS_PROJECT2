import shutil
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

def pdf_analysis_query(paths, model, prompt): 
    #model is to set what gpt model you want to use 
    #paths is an array of paths to documents
    load_dotenv()
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
    from langchain.vectorstores import Chroma

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

path = []
path.append('CHANGE TO YOUR DOC PATH')
model = 'gpt-4o'
prompt = "INPUT YOUR PROMPT"
print(pdf_analysis_query(path, model,prompt))