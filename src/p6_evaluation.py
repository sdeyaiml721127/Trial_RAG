# p6_evaluation.py
"""
Defines the RagasEvaluator class for RAG pipeline evaluation.
Refactored to use GenAI Lab (OpenAI) embeddings and LLM with Async support.
"""
from typing import List
import httpx

# Ragas imports
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from datasets import Dataset

# LangChain OpenAI imports
from langchain_openai import ChatOpenAI

# Import project modules
import p1_config as config
from p5_agent_service import AgentService
from p3_embeddings import get_openai_embeddings

class RagasEvaluator:
    """
    Handles the evaluation of the RAG pipeline using the RAGAs library.
    """
    def __init__(self):
        # 1. Setup Clients with SSL Bypass (Sync and Async)
        http_client = httpx.Client(verify=False)
        http_async_client = httpx.AsyncClient(verify=False) # <--- NEW ADDITION

        # 2. Create the LLM for RAGAs
        self.llm = ChatOpenAI(
            temperature=0,
            model=config.LLM_MODEL_NAME,
            base_url="https://genailab.tcs.in",
            api_key=config.GENAI_LLM_API_KEY,
            http_client=http_client,
            http_async_client=http_async_client # <--- REQUIRED FOR RAGAS
        )
        
        # 3. Create the Embeddings for RAGAs
        # (This now includes the async client from the p3 fix above)
        self.embeddings = get_openai_embeddings()
        
        # 4. Define Metrics
        self.metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy,
        ]
        
        print("RagasEvaluator initialized (using GenAI Lab with Async Support).")

    def evaluate_query(self, query: str, ground_truth: str, rag_answer: str, contexts: List[str]):
        """
        Evaluates a single query-answer pair against the RAGAs metrics.
        """
        if not rag_answer:
            print("Skipping evaluation due to empty RAG answer.")
            return None
        if not contexts:
            print("Skipping evaluation due to empty contexts.")
            return None

        # 1. Create the data dictionary
        data = {
            "question": [query],
            "ground_truth": [ground_truth],
            "answer": [rag_answer],
            "contexts": [contexts],
        }

        # 2. Convert to a Hugging Face Dataset
        dataset = Dataset.from_dict(data)

        # 3. Run the evaluation
        print("\n--- Running RAGAs Evaluation ---")
        try:
            # raise_exceptions=False prevents one metric failure from crashing the whole batch
            result = evaluate(
                dataset, 
                metrics=self.metrics, 
                llm=self.llm, 
                embeddings=self.embeddings,
                raise_exceptions=False 
            )
            
            print("--- RAGAs Evaluation Complete ---")
            return result
        except Exception as e:
            print(f"\n[RAGAs Evaluation Error]: {e}")
            return None

def run_evaluation_example(agent_service: AgentService):
    """
    Runs a single, pre-defined evaluation.
    """
    evaluator = RagasEvaluator()

    # --- Test Case ---
    eval_query = "explain steps in RAG Indexing."
    eval_ground_truth = (
        "The steps for RAG indexing are: "
        "1. Ingesting data from sources like PDFs. "
        "2. Chunking the documents into smaller segments. "
        "3. Embedding the chunks into vector representations. "
        "4. Storing the embeddings in a vector database."
    )
    # -----------------

    # 1. Get the RAG answer
    response = agent_service.run_query(eval_query)
    
    if 'output' not in response or not response['output']:
         print("Agent did not provide an output. Skipping evaluation.")
         return
         
    rag_answer = response['output']

    # 2. Get the retrieved contexts
    contexts = agent_service.last_retrieved_contexts

    # 3. Run evaluation
    eval_result = evaluator.evaluate_query(
        query=eval_query,
        ground_truth=eval_ground_truth,
        rag_answer=rag_answer,
        contexts=contexts
    )

    if eval_result:
        print("\n--- RAGAs Evaluation Results ---")
        print(eval_result)