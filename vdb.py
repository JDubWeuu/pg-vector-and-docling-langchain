from langchain_postgres import PGVector
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from hashlib import sha256
from langchain_core.prompts import PromptTemplate
from docling.chunking import HybridChunker
from langchain_groq import ChatGroq
from langchain_core.documents import Document
import os


# think about to make the db async later, cause you will need to for web servers


class VectorDatabase:
    def __init__(self, file_names: list[str] = [], database_link: str = "") -> None:
        if not database_link:
            raise ValueError("Need to provide a database link for the llm to refer to")
        self.embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2')
        self.documents = file_names
        self.vector_store = PGVector(
            embeddings=self.embeddings,
            collection_name="postgres vector db",
            use_jsonb=True,
            connection=database_link
        )
        self.retriever = self.vector_store.as_retriever(search_kwargs={
            "k": 3
        }, search_type="mmr")
        self.llm = ChatGroq(
            api_key=os.environ.get("GROQ_API_KEY"),
            model="gemma2-9b-it",
            max_tokens=None,
            temperature=0.5
        )
        self.prompt = PromptTemplate.from_template(template="""
        The person's resume information is below. Please make sure to read it thoroughly \n--------------\n
        {context}
        \n-----------------\n
        With the given person's resume and without using any prior knowledge, answer the query directly and concisely.\n
        Query: {input}\n
        Answer:\n
        """)
    
    def generate_doc_id(self, content: str):
        return sha256(content.encode()).hexdigest()
        
    def check_doc_exists(self, id):
        results: list[Document] = self.vector_store.similarity_search("", filter={
            "id": {
                "$in": [id]
            }
        })
        return True if results else False
    
    def chunk_docs(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        for i in range(len(self.documents)):
            # obtain full file path
            self.documents[i] = os.path.join(current_dir, "assets", self.documents[i])
        # pass all the files into the docling loader which will parse and chunk them
        d = DoclingLoader(chunker=HybridChunker(tokenizer="sentence-transformers/all-MiniLM-L6-v2"), file_path=self.documents, export_type=ExportType.DOC_CHUNKS)
        chunks: list[Document] = d.load()
        # existing_ids: set = self.get_document_ids()
        cleaned_chunks: list[Document] = []
        for doc in chunks:
            doc_id = self.generate_doc_id(doc.page_content)
            if not self.check_doc_exists(doc_id):
                doc.metadata["id"] = doc_id
                cleaned_chunks.append(doc)
        print(cleaned_chunks)
        if cleaned_chunks:
            self.vector_store.add_documents(cleaned_chunks, ids=[doc.metadata["id"] for doc in cleaned_chunks])
            print("Added new docs")
            return
        print("Did not add any new docs")
        
        
    """Query the LLM and the pdf for what you want within the resume"""
    def query(self, question: str) -> str:
        question_answer_chain = create_stuff_documents_chain(self.llm, self.prompt)
        rag_chain = create_retrieval_chain(self.retriever, question_answer_chain)
        response = rag_chain.invoke({
            "input": question
        })
        
        return response["answer"]
    


if __name__ == "__main__":
    vector_db = VectorDatabase(file_names=["vector_db_resume.pdf"], database_link=os.environ.get("DB_LINK"))
    vector_db.chunk_docs()
    print(vector_db.query("Why is this candidate a good fit for Walmart?"))