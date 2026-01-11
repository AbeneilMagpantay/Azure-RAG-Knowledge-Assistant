"""RAG evaluation metrics using RAGAS framework."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import time


@dataclass
class EvaluationResult:
    """Result of RAG evaluation."""
    
    question: str
    answer: str
    context: str
    ground_truth: Optional[str] = None
    
    # RAGAS metrics
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    
    # Additional metrics
    latency_ms: float = 0.0
    tokens_used: int = 0
    
    @property
    def overall_score(self) -> float:
        """Calculate overall score as weighted average."""
        weights = {
            "faithfulness": 0.3,
            "answer_relevancy": 0.3,
            "context_precision": 0.2,
            "context_recall": 0.2
        }
        return (
            self.faithfulness * weights["faithfulness"] +
            self.answer_relevancy * weights["answer_relevancy"] +
            self.context_precision * weights["context_precision"] +
            self.context_recall * weights["context_recall"]
        )


class RAGEvaluator:
    """
    Evaluate RAG pipeline quality using RAGAS metrics.
    
    Metrics:
    - Faithfulness: Is the answer grounded in the context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved docs relevant?
    - Context Recall: Are all relevant docs retrieved?
    """
    
    def __init__(
        self,
        llm_client=None,
        use_ragas: bool = True
    ):
        """
        Initialize evaluator.
        
        Args:
            llm_client: LLM client for LLM-based evaluation
            use_ragas: Use RAGAS library if available
        """
        self.llm_client = llm_client
        self.use_ragas = use_ragas
        self._ragas_available = self._check_ragas()
    
    def _check_ragas(self) -> bool:
        """Check if RAGAS is available."""
        try:
            import ragas
            return True
        except ImportError:
            return False
    
    def evaluate(
        self,
        question: str,
        answer: str,
        context: str,
        ground_truth: Optional[str] = None
    ) -> EvaluationResult:
        """
        Evaluate a single RAG response.
        
        Args:
            question: User question
            answer: Generated answer
            context: Retrieved context
            ground_truth: Optional ground truth answer
            
        Returns:
            EvaluationResult with metrics
        """
        result = EvaluationResult(
            question=question,
            answer=answer,
            context=context,
            ground_truth=ground_truth
        )
        
        # Time the evaluation
        start_time = time.time()
        
        if self.use_ragas and self._ragas_available:
            result = self._evaluate_with_ragas(result)
        else:
            result = self._evaluate_heuristic(result)
        
        result.latency_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _evaluate_with_ragas(self, result: EvaluationResult) -> EvaluationResult:
        """Evaluate using RAGAS library."""
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        )
        from datasets import Dataset
        
        # Prepare data for RAGAS
        data = {
            "question": [result.question],
            "answer": [result.answer],
            "contexts": [[result.context]],
        }
        
        if result.ground_truth:
            data["ground_truth"] = [result.ground_truth]
            metrics = [faithfulness, answer_relevancy, context_precision, context_recall]
        else:
            metrics = [faithfulness, answer_relevancy, context_precision]
        
        dataset = Dataset.from_dict(data)
        
        # Run evaluation
        eval_result = evaluate(dataset, metrics=metrics)
        
        result.faithfulness = eval_result.get("faithfulness", 0.0)
        result.answer_relevancy = eval_result.get("answer_relevancy", 0.0)
        result.context_precision = eval_result.get("context_precision", 0.0)
        result.context_recall = eval_result.get("context_recall", 0.0)
        
        return result
    
    def _evaluate_heuristic(self, result: EvaluationResult) -> EvaluationResult:
        """
        Evaluate using heuristic methods when RAGAS not available.
        Provides approximate scores based on text analysis.
        """
        # Faithfulness: Check if answer terms appear in context
        answer_words = set(result.answer.lower().split())
        context_words = set(result.context.lower().split())
        
        if answer_words:
            overlap = len(answer_words & context_words)
            result.faithfulness = min(overlap / len(answer_words), 1.0)
        
        # Answer relevancy: Check if question terms appear in answer
        question_words = set(result.question.lower().split())
        question_words -= {"what", "how", "why", "when", "where", "who", "is", "are", "the", "a", "an"}
        
        if question_words:
            overlap = len(question_words & answer_words)
            result.answer_relevancy = min(overlap / len(question_words), 1.0)
        
        # Context precision: Simple length ratio heuristic
        if result.context:
            # Assume shorter, focused context is more precise
            context_len = len(result.context)
            optimal_len = 2000  # Approximately optimal context length
            result.context_precision = min(optimal_len / max(context_len, 1), 1.0)
        
        # Context recall: If ground truth available, check coverage
        if result.ground_truth:
            truth_words = set(result.ground_truth.lower().split())
            if truth_words:
                overlap = len(truth_words & context_words)
                result.context_recall = min(overlap / len(truth_words), 1.0)
        
        return result
    
    def evaluate_batch(
        self,
        evaluations: List[Dict[str, Any]]
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple RAG responses.
        
        Args:
            evaluations: List of dicts with question, answer, context, ground_truth
            
        Returns:
            List of EvaluationResult
        """
        results = []
        for eval_data in evaluations:
            result = self.evaluate(
                question=eval_data["question"],
                answer=eval_data["answer"],
                context=eval_data["context"],
                ground_truth=eval_data.get("ground_truth")
            )
            results.append(result)
        
        return results
    
    def get_summary(self, results: List[EvaluationResult]) -> Dict[str, float]:
        """
        Get summary statistics for a batch of evaluations.
        
        Args:
            results: List of EvaluationResult
            
        Returns:
            Dict with average metrics
        """
        if not results:
            return {}
        
        n = len(results)
        return {
            "avg_faithfulness": sum(r.faithfulness for r in results) / n,
            "avg_answer_relevancy": sum(r.answer_relevancy for r in results) / n,
            "avg_context_precision": sum(r.context_precision for r in results) / n,
            "avg_context_recall": sum(r.context_recall for r in results) / n,
            "avg_overall_score": sum(r.overall_score for r in results) / n,
            "avg_latency_ms": sum(r.latency_ms for r in results) / n,
            "total_evaluations": n
        }
