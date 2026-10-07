"""
RAG Pipeline

main.py provides commands to run preprocesssing, execute RAG queries,
and evaluate pipeline performance.

Usage:

1. python3 main.py preprocess semantic --threshold 0.35
2. python3 main.py query --prompt "How are translations handled in FastAPI?"
3. python3 main.py eval --dataset eval_dataset.json

"""

import argparse
import json

from preprocessing.preprocessing_handler import PreprocessingPipe
from retrieval.retrieval import HybridRetrieval
from generation.generator import Generator
from generation.verifier import CitationVerifier
from evaluation.scorer import ConfidenceScore
from evaluation.guardrail import RetrievalGuardrail
from evaluation.evaluator import Evaluation


def run_preprocessing(strategy: str, strategy_kwargs: dict) -> None:
    """
    Executes document preprocessing and indexes chunks for retrieval.

    Applies the specified chunking strategy to raw documents, builds dense
    vector embeddings database and BM25 search indices.

    Args:
        strategy(str): The chunking algorithm ('fixed', 'recursive', 'semantic').
        strategy_kwargs (dict): Strategy-specific parameters.
    
    """

    print(f"Starting preprocessing, strategy: {strategy}")
    
    pipe = PreprocessingPipe()
    pipe.organiser(strategy_name=strategy, **strategy_kwargs)

    print("Preprocessing completed")

def run_query(query: str, key: str) -> dict:
    """
    Executes the full RAG pipeline for a given query.
    
    Pipeline:
        1. Hybrid Retrieval (BM25 + Dense Vector Search)
        2. Guardrail Interception Check (to stop low-confidence contexts early)
        3. Response Generation via LLM
        4. Citation Verification
        5. Composite Confidence Scoring
    
    Args:
        query(str): The user prompt or question
        key(str): The API key for LLM generation and citation verification
    
    Returns:
        dict: Pipeline output containing:
            - answer (str | dict): Generated text or guardrail fallback payload.
            - intercepted (bool, optional): True if blocked by retrieval guardrail.
            - confidence_results (dict | None): Composite score breakdown if generated.
            - verification_report (dict | None): Citation support report if generated.

    """

    retriever = HybridRetrieval()
    guardrail = RetrievalGuardrail(threshold=0.35)
    generator = Generator (key=key)
    verifier= CitationVerifier(key=key)
    scorer = ConfidenceScore() 


    #query = "How are translations handled in FastAPI?"
    
    # 1. Retrieval
    print("1. Retrieving context blocks")

    reranked_chunks = retriever.hybrid_search(query=query)

    # 2. Guardrail
    print("2. Checking retrieval scores before generation")

    guard_result = guardrail.check(reranked_chunks, query)

    if not guard_result["passed"]:

        print("LOW RETRIEVAL SCORE - STOPPING BEFORE GENERATION")
        print(guard_result["fallback_response"])

        return {
            "answer": guard_result["fallback_response"],
            "intercepted": True, #when too low
            "confidence_results": None,
            "verification_report": None
        }
    
    # 3. Generation
    print("3. Generating answer with Groq")

    answer = generator.generate(query=query, chunks=reranked_chunks)
    
    print("\n Generated Answer")
    print(answer)

    # 4. Citation verification
    print("4. Verifying Citations")

    verification_report = verifier.verify_all(answer, reranked_chunks)

    print(" \nVerification Report")
    print(f"Total citations checked: {verification_report['total_citations']}")
    print(f"Supported citations:     {verification_report['supported_count']}")
    print(f"Verification Score:      {verification_report['verification_score']:.1f}%")


    for v in verification_report["verifications"]:
        status = "PASSED" if v["is_supported"] else "FAILED"
        c_idx = v.get("citation_idx", "?")
        claim_text = v.get("claim", "")

        print(f' - [{status}] Chunk [{c_idx}]: "{claim_text}"')


    # 5. Evaluation. Confidence Scoring
    print("5. Calculating Confidence Scores")

    confidence_results = scorer.compute_scores(
        query=query,
        answer=answer,
        reranked_chunks=reranked_chunks,
        verification_report=verification_report
    )

    print("\nConfidence Report:")
    print(f"Composite Confidence Score: {confidence_results['composite_score'] * 100:.1f}%")
    print("Breakdown:")
    print(f"Retrieval Confidence: {confidence_results['breakdown']['retrieval_confidence'] * 100:.1f}%")
    print(f"Citation Coverage:    {confidence_results['breakdown']['citation_coverage'] * 100:.1f}%")
    print(f"Answer Completeness:  {confidence_results['breakdown']['answer_completeness'] * 100:.1f}%")

    return {
    "answer": answer,
    "confidence_results": confidence_results,
    "verification_report": verification_report
    }



def build_parser() -> argparse.ArgumentParser:
    """
    Constructs the argument parser for pipeline execution modes.

    Returns:
        argparse.ArgumentParser: Configured argument parser with subparsers for
        'preprocess', 'query', and 'eval' modes.
    """

    parser = argparse.ArgumentParser(description="RAG Pipeline")

    mode_subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode of operation")

    #Subcommand: Preprocess
    preprocess_parser = mode_subparsers.add_parser("preprocess", help="Chunk documents and build indexes")
    
    #Different chunking strategies and their parameters
    strategy_subparsers = preprocess_parser.add_subparsers(dest="strategy", required=True, help="Chunking strategy")

    fixed_parser = strategy_subparsers.add_parser("fixed")

    fixed_parser.add_argument("--size", type=int, default=500, help="Chunk size in characters")
    fixed_parser.add_argument("--overlap", type=int, default=50, help="Overlap in characters")

    rec_parser = strategy_subparsers.add_parser("recursive")
    rec_parser.add_argument("--size", type=int, default=500, help="Max chunk size in characters")

    sem_parser = strategy_subparsers.add_parser("semantic")
    sem_parser.add_argument("--threshold", type=float, default=0.35, help="Cosine similarity threshold")
    sem_parser.add_argument("--min_chunk_size", type=int, default=300, help="Minimum chunk size")

    #Subcommand: Query
    query_parser = mode_subparsers.add_parser("query", help="Ask a question")
    query_parser.add_argument("--prompt", type=str, required=True, help="Question to ask")

    #Subcommand: Evaluation 
    eval_parser = mode_subparsers.add_parser(
        "eval", help="Run benchmark evaluation across a dataset for eval with questions"
    )
    eval_parser.add_argument("--dataset", type=str, default="eval_dataset.json", help="Path to evaluation dataset")

    return parser

def main() -> None:
    """
    Main execution handles that parses commands and delegates execution.
    """
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "preprocess":
        args_dict = vars(args) #converts a python Namespace object into a python dict
        strategy_name = args_dict.pop("strategy")
        args_dict.pop("mode")

        run_preprocessing(strategy=strategy_name, strategy_kwargs=args_dict)
    
    elif args.mode == "query":
        with open("../keys.txt", "r") as f:
            key = f.read().strip()
        
        run_query(query=args.prompt, key=key)
    
    elif args.mode == "eval":
        evaluator = Evaluation()
        scores = evaluator.execute_evaluation(dataset_path=args.dataset)

        print("\nEvaluation Benchmark Results")
        print(json.dumps(scores, indent=2))

        if scores:
            avg_score = sum(item["score"] for item in scores) / len(scores)
            print(f"\nAverage Score: {avg_score:.2f} / 5.0")


    

if __name__== "__main__":
    main()