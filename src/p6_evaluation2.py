# p6_evaluation.py
"""
Defines the RagasEvaluator class for RAG pipeline evaluation.
Refactored to use GenAI Lab (OpenAI) embeddings and LLM with Async support.
"""
from typing import List
import httpx
import pandas as pd

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

def run_evaluation_batch(agent_service: AgentService):
    """
    Runs a batch of 4 pre-defined evaluations and prints a final summary table.
    """
    evaluator = RagasEvaluator()

    # Define the 4 Test Cases
    test_cases = [
        {
            "query": "What is the role of an Embedding Model in RAG?",
            "ground_truth": "The embedding model converts text data into numerical vectors. These vectors capture semantic meaning, allowing the system to perform similarity searches."
        },
        {
            "query": "Explain the concept of 'Chunking'.",
            "ground_truth": "Chunking is the process of breaking down large documents into smaller segments. It is necessary because LLMs have a maximum context window limit."
        },
        {
            "query": "How does RAG help reduce LLM hallucinations?",
            "ground_truth": "RAG reduces hallucinations by retrieving factual information from a verified knowledge base and forcing the LLM to use that context rather than training memory."
        },
        {
            "query": "Difference between Vector Search and Keyword Search?",
            "ground_truth": "Keyword search matches exact words. Vector search matches semantic meaning using embeddings, finding relevant results even without exact keyword matches."
        }
    ]

    results_data = []

    print(f"Starting Batch Evaluation of {len(test_cases)} queries...\n")

    for i, case in enumerate(test_cases):
        print(f"--- Processing Query {i+1}/{len(test_cases)}... ---")
        
        # 1. Get the RAG answer
        response = agent_service.run_query(case['query'])
        
        if 'output' not in response or not response['output']:
             print(f"Query {i+1} failed to generate output.")
             continue
             
        rag_answer = response['output']
        contexts = agent_service.last_retrieved_contexts

        # 2. Run evaluation
        eval_result = evaluator.evaluate_query(
            query=case['query'],
            ground_truth=case['ground_truth'],
            rag_answer=rag_answer,
            contexts=contexts
        )
        
        if eval_result:
            # 3. Extract metrics and add to list
            # FIX: Use square brackets [] instead of .get()
            row = {
                "Query": case['query'],
                "Context Precision": eval_result["context_precision"],
                "Context Recall": eval_result["context_recall"],
                "Faithfulness": eval_result["faithfulness"],
                "Answer Relevancy": eval_result["answer_relevancy"]
            }
            results_data.append(row)

    # 4. Create and Display Table
    if results_data:
        df = pd.DataFrame(results_data)
        
        # Formatting for cleaner output
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        pd.set_option('display.max_colwidth', 50) 

        print("\n\n========== FINAL EVALUATION SUMMARY ==========")
        print(df)
    else:
        print("No results were generated.")