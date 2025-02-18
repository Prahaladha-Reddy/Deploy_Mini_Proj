from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from typing import List, Any


class RetrievalSystem:
    def __init__(self,embeddings_model):
        """
        Initialize the retrieval system with embedding model and search parameters.
        
        Args:
            embedding_model: Name of the HuggingFace embedding model
            model_kwargs: Arguments for the embedding model
            search_kwargs: Retrieval parameters (e.g., number of results to return)
        """

        self.embedding_model = embeddings_model
        self.search_kwargs = {"k": 3}
        self.vector_store = None
        self.retriever = None

    def build_vector_store(self, chunks: List[str]) -> None:
        """
        Create FAISS vector store from text chunks.
        
        Args:
            chunks: List of text documents to index
        """
        # Convert text chunks to Document objects
        
        # Create FAISS vector store
        self.vector_store = FAISS.from_documents(
            documents=chunks,
            embedding=self.embedding_model
        )
        
        # Create retriever
        self.retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs=self.search_kwargs
        )

        return self.retriever

    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents relevant to the query.
        
        Args:
            query: Search query string
            
        Returns:
            List of relevant documents
        """
        if not self.retriever:
            raise ValueError("Vector store not initialized. Call build_vector_store first.")
            
        return self.retriever.invoke(query)

    def save_index(self, path: str) -> None:
        """Save FAISS index to disk"""
        if self.vector_store:
            self.vector_store.save_local(path)

    @classmethod
    def load_index(cls, path: str, embedding_model: Any) -> 'RetrievalSystem':
        """
        Load FAISS index from disk
        
        Args:
            path: Path to saved index
            embedding_model: Embedding model instance
            
        Returns:
            RetrievalSystem instance
        """
        instance = cls()
        instance.vector_store = FAISS.load_local(path, embedding_model)
        instance.retriever = instance.vector_store.as_retriever()
        return instance