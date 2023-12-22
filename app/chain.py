# Load
from langchain.chat_models import ChatOllama
from langchain.embeddings import GPT4AllEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

from model import model
from vectorstore import vectorstore
from retriever import CustomRetriever

k_samples = 5
custom_retriever = CustomRetriever(retriever = vectorstore.as_retriever(search_kwargs={"k": k_samples}))

# Prompt
from langchain import PromptTemplate

template = """作為一個美食推薦家,你需要使用位於context中的內容去回答問題,且遵循rule中的規則。

rule
1.只要輸出簡潔的一句話就好
2.盡量不要輸出距離有多遠除非問題有提及遠近相關的詞彙
3.不要輸出ASSISTANT: 
endrule

context
{context}
endcontext

問題: {question}
回答: """

prompt = PromptTemplate(input_variables=["context", "question"], template=template)

# RAG chain
chain = (
    RunnableParallel({"context": custom_retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
