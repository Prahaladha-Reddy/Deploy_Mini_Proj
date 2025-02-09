from retrieval import RetrievalSystem


from langchain_groq import ChatGroq
import os 
import dotenv

dotenv.load_dotenv()

os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain




class RAGPipeline:
    def __init__(self, retrieval_system: RetrievalSystem):
        self.retrieval_system = retrieval_system
        self.llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0)
        self.prompt_template = self._create_prompt_template()
        self.llm_chain = self._create_llm_chain()

    def _create_prompt_template(self) -> PromptTemplate:
        prompt = """
        You are an expert AI assistant. Explain the answer to the question using the provided context.

        Context:
        {context}

        Question:
        {question}

        Provide a clear, detailed answer:
        """
        return PromptTemplate(
            template=prompt,
            input_variables=["context", "question"]
        )

    def _create_llm_chain(self) -> LLMChain:
        return LLMChain(
            prompt=self.prompt_template,
            llm=self.llm
        )

    def generate_answer(self, question: str) -> str:
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retrieval_system.retrieve(question)
        
        # Step 2: Format the context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Step 3: Generate answer using both context and original question
        return self.llm_chain.run(context=context, question=question)