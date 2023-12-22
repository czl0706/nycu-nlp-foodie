from langchain.schema.retriever import BaseRetriever
from langchain_core.documents import Document
from typing import List
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun, AsyncCallbackManagerForRetrieverRun
import random

# Retriever with random sampling
class CustomRetriever(BaseRetriever):
    retriever: BaseRetriever
        
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        # Use your existing retriever to get the documents
        documents = self.retriever.get_relevant_documents(query, callbacks=run_manager.get_child())
        
        # print(documents)
        
        # Sort the documents by "source"
        # documents = sorted(documents, key=lambda doc: doc.metadata.get('source'))
        
        return [random.choice(documents)]
        # return (*random.sample(documents, 2), )
        
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        # **kwargs: Any,
    ) -> List[Document]:
        documents = await self.retriever.aget_relevant_documents(query, callbacks=run_manager.get_child())
        return [random.choice(documents)]