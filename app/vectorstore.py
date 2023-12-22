from langchain.document_loaders import CSVLoader
loader = CSVLoader('summary_results.csv')
data = loader.load()

# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024,
                                               chunk_overlap  = 100,
                                               length_function = len,
                                               is_separator_regex = False)

all_splits = text_splitter.split_documents(data)

# Add to vectorDB
from langchain.embeddings import HuggingFaceEmbeddings 
embeddings = HuggingFaceEmbeddings(model_name='shibing624/text2vec-base-chinese', 
                                   model_kwargs={'device': 'cuda'}) 

from langchain.vectorstores import FAISS
vectorstore = FAISS.from_documents(data, embeddings)