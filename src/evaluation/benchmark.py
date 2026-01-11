"""Benchmark runner for RAG pipeline evaluation."""

import json
import csv
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

from .metrics import RAGEvaluator, EvaluationResult


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark run."""
    
    name: str = "default_benchmark"
    description: str = ""
    top_k: int = 5
    use_multi_query: bool = False
    search_type: str = "hybrid"


@dataclass
class BenchmarkResult:
    """Result of a benchmark run."""
    
    config: BenchmarkConfig
    results: List[EvaluationResult]
    summary: Dict[str, float]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()


class BenchmarkRunner:
    """
    Run benchmarks on RAG pipeline.
    
    Features:
    - Load test questions from JSON/CSV
    - Run RAG pipeline and evaluate
    - Export results to JSON/CSV
    - Compare multiple configurations
    """
    
    def __init__(
        self,
        rag_chain,
        evaluator: Optional[RAGEvaluator] = None,
        output_dir: str = "./benchmark_results"
    ):
        """
        Initialize benchmark runner.
        
        Args:
            rag_chain: RAGChain instance to benchmark
            evaluator: RAGEvaluator instance
            output_dir: Directory for output files
        """
        self.rag_chain = rag_chain
        self.evaluator = evaluator or RAGEvaluator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_questions(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load test questions from file.
        
        Expected format:
        JSON: [{"question": "...", "ground_truth": "..."}, ...]
        CSV: question,ground_truth columns
        
        Args:
            filepath: Path to questions file
            
        Returns:
            List of question dicts
        """
        path = Path(filepath)
        
        if path.suffix == ".json":
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        elif path.suffix == ".csv":
            questions = []
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    questions.append({
                        "question": row.get("question", ""),
                        "ground_truth": row.get("ground_truth", "")
                    })
            return questions
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")
    
    def run(
        self,
        questions: List[Dict[str, Any]],
        config: Optional[BenchmarkConfig] = None,
        show_progress: bool = True
    ) -> BenchmarkResult:
        """
        Run benchmark on a set of questions.
        
        Args:
            questions: List of question dicts
            config: Benchmark configuration
            show_progress: Show progress bar
            
        Returns:
            BenchmarkResult
        """
        config = config or BenchmarkConfig()
        results = []
        
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(questions, desc=f"Running {config.name}")
            except ImportError:
                iterator = questions
        else:
            iterator = questions
        
        for q_data in iterator:
            question = q_data["question"]
            ground_truth = q_data.get("ground_truth")
            
            # Run RAG pipeline
            response = self.rag_chain.query(
                question=question,
                top_k=config.top_k,
                use_multi_query=config.use_multi_query
            )
            
            # Evaluate
            eval_result = self.evaluator.evaluate(
                question=question,
                answer=response.answer,
                context=response.context_used,
                ground_truth=ground_truth
            )
            eval_result.tokens_used = response.tokens_used
            
            results.append(eval_result)
        
        # Calculate summary
        summary = self.evaluator.get_summary(results)
        
        return BenchmarkResult(
            config=config,
            results=results,
            summary=summary
        )
    
    def run_from_file(
        self,
        filepath: str,
        config: Optional[BenchmarkConfig] = None
    ) -> BenchmarkResult:
        """Run benchmark from questions file."""
        questions = self.load_questions(filepath)
        return self.run(questions, config)
    
    def export_results(
        self,
        result: BenchmarkResult,
        format: str = "json"
    ) -> str:
        """
        Export benchmark results to file.
        
        Args:
            result: BenchmarkResult to export
            format: "json" or "csv"
            
        Returns:
            Path to exported file
        """
        filename = f"{result.config.name}_{result.timestamp.replace(':', '-')}"
        
        if format == "json":
            filepath = self.output_dir / f"{filename}.json"
            
            # Convert to serializable format
            export_data = {
                "config": asdict(result.config),
                "summary": result.summary,
                "timestamp": result.timestamp,
                "results": [
                    {
                        "question": r.question,
                        "answer": r.answer,
                        "faithfulness": r.faithfulness,
                        "answer_relevancy": r.answer_relevancy,
                        "context_precision": r.context_precision,
                        "context_recall": r.context_recall,
                        "overall_score": r.overall_score,
                        "latency_ms": r.latency_ms,
                        "tokens_used": r.tokens_used
                    }
                    for r in result.results
                ]
            }
            
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2)
        
        elif format == "csv":
            filepath = self.output_dir / f"{filename}.csv"
            
            with open(filepath, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    "question", "answer", "faithfulness", "answer_relevancy",
                    "context_precision", "context_recall", "overall_score",
                    "latency_ms", "tokens_used"
                ])
                
                # Data
                for r in result.results:
                    writer.writerow([
                        r.question, r.answer[:200], r.faithfulness,
                        r.answer_relevancy, r.context_precision, r.context_recall,
                        r.overall_score, r.latency_ms, r.tokens_used
                    ])
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        print(f"Exported results to: {filepath}")
        return str(filepath)
    
    def compare_configs(
        self,
        questions: List[Dict[str, Any]],
        configs: List[BenchmarkConfig]
    ) -> Dict[str, BenchmarkResult]:
        """
        Compare multiple configurations.
        
        Args:
            questions: Test questions
            configs: List of configurations to compare
            
        Returns:
            Dict mapping config name to BenchmarkResult
        """
        results = {}
        
        for config in configs:
            print(f"\nRunning config: {config.name}")
            result = self.run(questions, config)
            results[config.name] = result
            
            print(f"  Overall score: {result.summary['avg_overall_score']:.3f}")
            print(f"  Faithfulness: {result.summary['avg_faithfulness']:.3f}")
            print(f"  Avg latency: {result.summary['avg_latency_ms']:.1f}ms")
        
        return results


def create_sample_questions() -> List[Dict[str, Any]]:
    """Create sample questions for testing."""
    return [
        {
            "question": "What is Azure OpenAI?",
            "ground_truth": "Azure OpenAI is a cloud service that provides access to OpenAI's language models through Microsoft Azure."
        },
        {
            "question": "How do I create an Azure AI Search index?",
            "ground_truth": "You can create an Azure AI Search index using the Azure portal, REST API, or SDK."
        },
        {
            "question": "What is RAG in AI?",
            "ground_truth": "RAG (Retrieval-Augmented Generation) is a technique that combines document retrieval with language model generation to provide grounded answers."
        }
    ]
