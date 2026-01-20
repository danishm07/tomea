"""
ML Experiment Result Analyzer - Analyze experiments and generate intelligent recommendations.

This module provides tools for comparing multiple ML methods across dimensions
(performance, efficiency, complexity) and generating context-aware recommendations
using LLM-powered analysis.

Example usage:
    >>> analyzer = ResultAnalyzer(llm_client=deepseek_client)
    >>> results = [ExperimentResult(method="LoRA", accuracy=0.871, ...)]
    >>> profile = DatasetProfile(task_type="classification", ...)
    >>> recommendation = analyzer.analyze_and_recommend(results, profile, user_context)
    >>> print(recommendation.full_report)

Dependencies:
    - pandas: For data manipulation and table formatting
    - json: For parsing LLM responses
    - LLM client (OpenAI/LangChain compatible)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, Protocol, Dict, List

logger = logging.getLogger(__name__)


class AnalyzerError(Exception):
    """Base exception for analyzer errors."""
    pass


class LLMError(AnalyzerError):
    """Error during LLM interaction."""
    pass


class ValidationError(AnalyzerError):
    """Error during data validation."""
    pass


class TaskType(str, Enum):
    """Supported ML task types."""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    SEQUENCE_LABELING = "sequence_labeling"
    GENERATION = "generation"


class SequenceVariance(str, Enum):
    """Sequence length variance categories."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class LabelBalance(str, Enum):
    """Label distribution categories."""
    BALANCED = "balanced"
    SLIGHTLY_IMBALANCED = "slightly_imbalanced"
    IMBALANCED = "imbalanced"
    HIGHLY_IMBALANCED = "highly_imbalanced"


class LLMClient(Protocol):
    """Protocol for LLM client compatibility."""

    def generate(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        ...


@dataclass
class ExperimentResult:
    """
    Results from a single ML experiment.

    Attributes:
        method: Name of the method/approach (e.g., "LoRA", "Gated Attention")
        accuracy: Classification accuracy (0.0-1.0)
        f1: F1 score (0.0-1.0)
        training_time: Training duration in hours
        memory_peak: Peak memory usage in GB
        parameters: Number of trainable parameters in millions
        status: Experiment status ('success' or 'failed')
        error: Error message if status is 'failed'
        precision: Optional precision score
        recall: Optional recall score
        inference_time: Optional inference time per sample (ms)
        metadata: Additional experiment metadata
    """

    method: str
    accuracy: float
    f1: float
    training_time: float
    memory_peak: float
    parameters: float
    status: str = "success"
    error: Optional[str] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    inference_time: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate result data after initialization."""
        if self.status not in ("success", "failed"):
            raise ValidationError(f"Invalid status: {self.status}")

        if self.status == "success":
            if not 0 <= self.accuracy <= 1:
                raise ValidationError(f"Accuracy must be 0-1, got {self.accuracy}")
            if not 0 <= self.f1 <= 1:
                raise ValidationError(f"F1 must be 0-1, got {self.f1}")

    @property
    def is_successful(self) -> bool:
        """Check if experiment completed successfully."""
        return self.status == "success"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "method": self.method,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "training_time": self.training_time,
            "memory_peak": self.memory_peak,
            "parameters": self.parameters,
            "inference_time": self.inference_time,
            "status": self.status,
            "error": self.error,
        }


@dataclass
class DatasetProfile:
    """
    Profile of a dataset's characteristics.

    Attributes:
        task_type: Type of ML task
        num_samples: Total number of samples
        num_classes: Number of classes (for classification)
        avg_sequence_length: Average sequence length in tokens
        sequence_length_variance: Variance category (low/medium/high)
        label_balance: Label distribution category
        domain: Dataset domain (medical, finance, etc.)
        max_sequence_length: Maximum observed sequence length
        min_sequence_length: Minimum observed sequence length
        long_sequence_ratio: Ratio of sequences > 400 tokens
        metadata: Additional dataset metadata
    """

    task_type: str
    num_samples: int
    avg_sequence_length: float
    sequence_length_variance: str = "medium"
    label_balance: str = "balanced"
    domain: str = "general"
    num_classes: Optional[int] = None
    max_sequence_length: Optional[int] = None
    min_sequence_length: Optional[int] = None
    long_sequence_ratio: Optional[float] = None
    class_distribution: Optional[dict[str, int]] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate profile data."""
        if self.num_samples <= 0:
            raise ValidationError(f"num_samples must be positive, got {self.num_samples}")

    @property
    def is_small_dataset(self) -> bool:
        """Check if dataset is considered small (<1000 samples)."""
        return self.num_samples < 1000

    @property
    def is_large_dataset(self) -> bool:
        """Check if dataset is considered large (>100k samples)."""
        return self.num_samples > 100_000

    @property
    def has_long_sequences(self) -> bool:
        """Check if dataset has significant long sequences."""
        if self.long_sequence_ratio is not None:
            return self.long_sequence_ratio > 0.2
        return self.avg_sequence_length > 300

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "task_type": self.task_type,
            "num_samples": self.num_samples,
            "num_classes": self.num_classes,
            "avg_sequence_length": self.avg_sequence_length,
            "sequence_length_variance": self.sequence_length_variance,
            "label_balance": self.label_balance,
            "domain": self.domain,
            "max_sequence_length": self.max_sequence_length,
            "min_sequence_length": self.min_sequence_length,
            "long_sequence_ratio": self.long_sequence_ratio,
        }


@dataclass
class Recommendation:
    """
    ML method recommendation with reasoning.

    Attributes:
        recommended_method: Name of the recommended method
        reasoning: List of reasons why this method is recommended
        tradeoffs: List of tradeoffs/warnings to consider
        confidence: Confidence score (0.0-1.0)
        performance_gain: Expected performance improvement over baseline (%)
        full_report: Complete markdown report
        comparison_data: Raw comparison data used for recommendation
        alternative_methods: Other viable methods in order of preference
    """

    recommended_method: str
    reasoning: list[str]
    tradeoffs: list[str]
    confidence: float
    performance_gain: float
    full_report: str
    comparison_data: dict[str, Any] = field(default_factory=dict)
    alternative_methods: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate recommendation data."""
        if not 0 <= self.confidence <= 1:
            raise ValidationError(f"Confidence must be 0-1, got {self.confidence}")

    def summary(self) -> str:
        """Return a brief summary of the recommendation."""
        return (
            f"Recommended: {self.recommended_method} "
            f"(+{self.performance_gain:.1f}% gain, {self.confidence:.0%} confidence)"
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "recommended_method": self.recommended_method,
            "reasoning": self.reasoning,
            "tradeoffs": self.tradeoffs,
            "confidence": self.confidence,
            "performance_gain": self.performance_gain,
            "alternative_methods": self.alternative_methods,
        }


class ResultAnalyzer:
    """
    Analyzes ML experiment results and generates intelligent recommendations.

    The analyzer compares multiple ML methods across performance, efficiency,
    and complexity dimensions, then uses an LLM to generate context-aware
    recommendations with detailed reasoning.

    Attributes:
        llm: LLM client for generating analysis
        baseline_method: Name of the baseline method for comparison

    Example:
        >>> analyzer = ResultAnalyzer(llm_client=deepseek_client)
        >>> results = [
        ...     ExperimentResult(method="Baseline", accuracy=0.854, ...),
        ...     ExperimentResult(method="LoRA", accuracy=0.871, ...),
        ...     ExperimentResult(method="Gated Attention", accuracy=0.913, ...),
        ... ]
        >>> profile = DatasetProfile(
        ...     task_type="classification",
        ...     num_samples=10000,
        ...     num_classes=5,
        ...     avg_sequence_length=256,
        ... )
        >>> recommendation = analyzer.analyze_and_recommend(
        ...     results, profile, {"max_training_time": 4.0}
        ... )
    """

    # Default weights for multi-criteria scoring
    DEFAULT_WEIGHTS = {
        "accuracy": 0.30,
        "f1": 0.25,
        "training_time": 0.15,
        "memory": 0.10,
        "parameters": 0.10,
        "inference_time": 0.10,
    }

    # Domain-specific weight adjustments
    DOMAIN_WEIGHTS = {
        "medical": {"accuracy": 0.40, "f1": 0.30},  # Prioritize accuracy
        "finance": {"inference_time": 0.20, "accuracy": 0.35},  # Latency matters
        "real_time": {"inference_time": 0.30, "memory": 0.15},  # Speed critical
    }

    def __init__(
        self,
        llm_client: Any = None,
        baseline_method: str = "baseline",
        weights: Optional[dict[str, float]] = None,
    ):
        """
        Initialize the ResultAnalyzer.

        Args:
            llm_client: LLM client for generating recommendations
            baseline_method: Name of baseline method for comparison
            weights: Custom weights for multi-criteria scoring
        """
        self.llm = llm_client
        self.baseline_method = baseline_method.lower()
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()

    def analyze_and_recommend(
        self,
        results: list[ExperimentResult],
        dataset_profile: DatasetProfile,
        user_context: Optional[dict[str, Any]] = None,
    ) -> Recommendation:
        """
        Analyze experiment results and return a recommendation.

        This is the main entry point. It compares results across dimensions,
        matches methods to dataset characteristics, and generates an
        LLM-powered recommendation with reasoning.

        Args:
            results: List of experiment results to compare
            dataset_profile: Profile of the target dataset
            user_context: User constraints and preferences

        Returns:
            Recommendation with reasoning and full report

        Raises:
            ValidationError: If results are invalid or insufficient
            LLMError: If LLM generation fails
        """
        user_context = user_context or {}

        # Validate inputs
        self._validate_results(results)

        # Filter to successful experiments
        successful_results = [r for r in results if r.is_successful]
        if not successful_results:
            raise ValidationError("No successful experiments to analyze")

        logger.info(f"Analyzing {len(successful_results)} successful experiments")

        # Compare results across dimensions
        comparison = self._compare_results(successful_results)

        # Analyze method-dataset fit
        fit_analysis = self._match_to_dataset(comparison, dataset_profile)

        # Adjust weights based on domain
        adjusted_weights = self._adjust_weights_for_context(
            dataset_profile, user_context
        )

        # Calculate composite scores
        scores = self._calculate_composite_scores(
            successful_results, adjusted_weights, user_context
        )

        # Combine all analysis
        full_analysis = {
            "comparison": comparison,
            "fit_analysis": fit_analysis,
            "scores": scores,
            "dataset": dataset_profile.to_dict(),
            "user_context": user_context,
        }

        # Generate recommendation (with or without LLM)
        if self.llm:
            recommendation = self._generate_recommendation_with_llm(
                full_analysis, successful_results, dataset_profile, user_context
            )
        else:
            recommendation = self._generate_recommendation_heuristic(
                full_analysis, successful_results, dataset_profile
            )

        # Format the full report
        recommendation.full_report = self._format_report(
            recommendation, successful_results, dataset_profile, comparison
        )

        return recommendation

    def _validate_results(self, results: list[ExperimentResult]) -> None:
        """Validate experiment results."""
        if not results:
            raise ValidationError("No experiment results provided")

        methods = [r.method for r in results]
        if len(methods) != len(set(methods)):
            raise ValidationError("Duplicate method names found")

    def _compare_results(
        self, results: list[ExperimentResult]
    ) -> dict[str, Any]:
        """
        Compare experiment results across all dimensions.

        Args:
            results: List of successful experiment results

        Returns:
            Dictionary containing comparison metrics and rankings
        """
        comparison = {
            "methods": [r.method for r in results],
            "metrics": {},
            "rankings": {},
            "normalized": {},
        }

        # Extract metrics
        metrics_to_compare = [
            ("accuracy", True),  # (metric_name, higher_is_better)
            ("f1", True),
            ("precision", True),
            ("recall", True),
            ("training_time", False),
            ("memory_peak", False),
            ("parameters", False),
            ("inference_time", False),
        ]

        for metric_name, higher_is_better in metrics_to_compare:
            values = []
            for r in results:
                val = getattr(r, metric_name, None)
                values.append(val if val is not None else float("nan"))

            # Filter out NaN for statistics
            valid_values = [v for v in values if v == v]  # NaN != NaN

            if valid_values:
                comparison["metrics"][metric_name] = {
                    "values": dict(zip(comparison["methods"], values)),
                    "min": min(valid_values),
                    "max": max(valid_values),
                    "mean": sum(valid_values) / len(valid_values),
                    "higher_is_better": higher_is_better,
                }

                # Rank methods (1 = best)
                sorted_methods = sorted(
                    [(m, v) for m, v in zip(comparison["methods"], values) if v == v],
                    key=lambda x: x[1],
                    reverse=higher_is_better,
                )
                comparison["rankings"][metric_name] = {
                    m: rank + 1 for rank, (m, _) in enumerate(sorted_methods)
                }

                # Normalize values to 0-1 (1 = best)
                min_val, max_val = min(valid_values), max(valid_values)
                if max_val > min_val:
                    normalized = {}
                    for m, v in zip(comparison["methods"], values):
                        if v == v:  # Not NaN
                            norm = (v - min_val) / (max_val - min_val)
                            normalized[m] = norm if higher_is_better else (1 - norm)
                        else:
                            normalized[m] = 0.0
                    comparison["normalized"][metric_name] = normalized

        # Find baseline for relative comparisons
        baseline_result = None
        for r in results:
            if r.method.lower() == self.baseline_method:
                baseline_result = r
                break

        if baseline_result:
            comparison["relative_to_baseline"] = {}
            for r in results:
                if r.method.lower() != self.baseline_method:
                    comparison["relative_to_baseline"][r.method] = {
                        "accuracy_delta": r.accuracy - baseline_result.accuracy,
                        "accuracy_pct_gain": (
                            (r.accuracy - baseline_result.accuracy)
                            / baseline_result.accuracy
                            * 100
                        ),
                        "f1_delta": r.f1 - baseline_result.f1,
                        "speedup": (
                            baseline_result.training_time / r.training_time
                            if r.training_time > 0
                            else float("inf")
                        ),
                    }

        return comparison

    def _match_to_dataset(
        self, comparison: dict[str, Any], dataset: DatasetProfile
    ) -> dict[str, Any]:
        """
        Analyze how well each method fits the dataset characteristics.

        Args:
            comparison: Comparison metrics from _compare_results
            dataset: Dataset profile

        Returns:
            Dictionary with fit scores and analysis per method
        """
        fit_analysis = {"methods": {}, "dataset_challenges": []}

        # Identify dataset challenges
        if dataset.has_long_sequences:
            fit_analysis["dataset_challenges"].append("long_sequences")
        if dataset.is_small_dataset:
            fit_analysis["dataset_challenges"].append("small_data")
        if dataset.label_balance in ("imbalanced", "highly_imbalanced"):
            fit_analysis["dataset_challenges"].append("imbalanced_labels")
        if dataset.sequence_length_variance == "high":
            fit_analysis["dataset_challenges"].append("variable_length")

        # Method-specific fit analysis (heuristics based on common patterns)
        method_characteristics = {
            "lora": {
                "strengths": ["parameter_efficient", "fast_training", "low_memory"],
                "weaknesses": ["long_sequences"],
                "best_for": ["small_data", "quick_iteration"],
            },
            "gated attention": {
                "strengths": ["long_sequences", "variable_length"],
                "weaknesses": ["high_memory"],
                "best_for": ["long_sequences", "complex_patterns"],
            },
            "adapter": {
                "strengths": ["parameter_efficient", "modular"],
                "weaknesses": [],
                "best_for": ["multi_task", "small_data"],
            },
            "full finetune": {
                "strengths": ["maximum_performance"],
                "weaknesses": ["high_memory", "slow_training", "overfitting_risk"],
                "best_for": ["large_data"],
            },
            "prefix tuning": {
                "strengths": ["very_parameter_efficient"],
                "weaknesses": ["limited_capacity"],
                "best_for": ["small_data", "quick_experiments"],
            },
        }

        for method in comparison["methods"]:
            method_lower = method.lower()
            fit_analysis["methods"][method] = {
                "fit_score": 0.5,  # Default neutral
                "pros": [],
                "cons": [],
            }

            # Check against known method characteristics
            for known_method, chars in method_characteristics.items():
                if known_method in method_lower:
                    # Calculate fit based on overlap with dataset challenges
                    pros = []
                    cons = []

                    for challenge in fit_analysis["dataset_challenges"]:
                        if challenge in chars.get("best_for", []):
                            pros.append(f"Good for {challenge.replace('_', ' ')}")
                        if challenge in chars.get("weaknesses", []):
                            cons.append(f"May struggle with {challenge.replace('_', ' ')}")

                    for strength in chars.get("strengths", []):
                        pros.append(strength.replace("_", " ").title())

                    fit_analysis["methods"][method]["pros"] = pros
                    fit_analysis["methods"][method]["cons"] = cons

                    # Simple fit score
                    fit_score = 0.5 + 0.1 * len(pros) - 0.1 * len(cons)
                    fit_analysis["methods"][method]["fit_score"] = max(0, min(1, fit_score))

        return fit_analysis

    def _adjust_weights_for_context(
        self, dataset: DatasetProfile, user_context: dict[str, Any]
    ) -> dict[str, float]:
        """Adjust scoring weights based on dataset and user context."""
        weights = self.weights.copy()

        # Adjust for domain
        if dataset.domain in self.DOMAIN_WEIGHTS:
            for metric, weight in self.DOMAIN_WEIGHTS[dataset.domain].items():
                weights[metric] = weight

        # Adjust for user constraints
        if user_context.get("prioritize_speed"):
            weights["training_time"] = 0.25
            weights["inference_time"] = 0.20
        if user_context.get("memory_constrained"):
            weights["memory"] = 0.25
        if user_context.get("prioritize_accuracy"):
            weights["accuracy"] = 0.40
            weights["f1"] = 0.30

        # Normalize weights to sum to 1
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}

    def _calculate_composite_scores(
        self,
        results: list[ExperimentResult],
        weights: dict[str, float],
        user_context: dict[str, Any],
    ) -> dict[str, float]:
        """
        Calculate weighted composite scores for each method.

        Args:
            results: Experiment results
            weights: Metric weights
            user_context: User constraints

        Returns:
            Dictionary mapping method names to composite scores
        """
        scores = {}

        # Find min/max for normalization
        metrics_data = {}
        for metric in ["accuracy", "f1", "training_time", "memory_peak", "parameters"]:
            values = [getattr(r, metric, None) for r in results]
            valid = [v for v in values if v is not None]
            if valid:
                metrics_data[metric] = {
                    "min": min(valid),
                    "max": max(valid),
                    "higher_is_better": metric in ("accuracy", "f1"),
                }

        for result in results:
            score = 0.0

            for metric, weight in weights.items():
                # Map weight keys to result attributes
                attr_name = metric
                if metric == "memory":
                    attr_name = "memory_peak"

                val = getattr(result, attr_name, None)
                if val is None or metric not in metrics_data:
                    continue

                data = metrics_data.get(attr_name, metrics_data.get(metric))
                if data is None:
                    continue

                # Normalize
                min_val, max_val = data["min"], data["max"]
                if max_val > min_val:
                    normalized = (val - min_val) / (max_val - min_val)
                    if not data["higher_is_better"]:
                        normalized = 1 - normalized
                else:
                    normalized = 1.0

                score += weight * normalized

            # Apply constraint penalties
            if user_context.get("max_training_time"):
                if result.training_time > user_context["max_training_time"]:
                    score *= 0.5  # Heavy penalty
            if user_context.get("max_memory"):
                if result.memory_peak > user_context["max_memory"]:
                    score *= 0.5

            scores[result.method] = score

        return scores

    def _generate_recommendation_with_llm(
        self,
        analysis: dict[str, Any],
        results: list[ExperimentResult],
        dataset: DatasetProfile,
        user_context: dict[str, Any],
    ) -> Recommendation:
        """Generate recommendation using LLM."""
        prompt = self._build_llm_prompt(results, dataset, user_context, analysis)

        try:
            response = self._call_llm(prompt)
            parsed = self._parse_llm_response(response)

            return Recommendation(
                recommended_method=parsed["recommended_method"],
                reasoning=parsed.get("reasoning", []),
                tradeoffs=parsed.get("tradeoffs", []),
                confidence=parsed.get("confidence", 0.7),
                performance_gain=parsed.get("performance_gain_pct", 0.0),
                full_report="",  # Will be filled later
                comparison_data=analysis,
                alternative_methods=parsed.get("alternatives", []),
            )
        except Exception as e:
            logger.warning(f"LLM recommendation failed: {e}, falling back to heuristic")
            return self._generate_recommendation_heuristic(analysis, results, dataset)

    def _generate_recommendation_heuristic(
        self,
        analysis: dict[str, Any],
        results: list[ExperimentResult],
        dataset: DatasetProfile,
    ) -> Recommendation:
        """Generate recommendation using heuristics (fallback)."""
        scores = analysis["scores"]
        comparison = analysis["comparison"]

        # Find best method by composite score
        best_method = max(scores, key=scores.get)
        best_result = next(r for r in results if r.method == best_method)

        # Find baseline for comparison
        baseline_result = next(
            (r for r in results if r.method.lower() == self.baseline_method),
            results[0],
        )

        # Calculate performance gain
        if baseline_result.method != best_method:
            perf_gain = (
                (best_result.accuracy - baseline_result.accuracy)
                / baseline_result.accuracy
                * 100
            )
        else:
            perf_gain = 0.0

        # Generate reasoning
        reasoning = []
        if best_result.accuracy == max(r.accuracy for r in results):
            reasoning.append(
                f"Highest accuracy: {best_result.accuracy:.1%}"
            )
        if best_result.f1 == max(r.f1 for r in results):
            reasoning.append(f"Best F1 score: {best_result.f1:.1%}")
        if best_result.training_time == min(r.training_time for r in results):
            reasoning.append(
                f"Fastest training: {best_result.training_time:.1f} hours"
            )

        # Generate tradeoffs
        tradeoffs = []
        fit_info = analysis.get("fit_analysis", {}).get("methods", {}).get(best_method, {})
        tradeoffs.extend(fit_info.get("cons", []))

        if best_result.memory_peak == max(r.memory_peak for r in results):
            tradeoffs.append(f"Highest memory usage: {best_result.memory_peak:.1f}GB")

        # Calculate confidence based on score margin
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and sorted_scores[0] > 0:
            margin = (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
            confidence = min(0.95, 0.6 + margin)
        else:
            confidence = 0.7

        # Get alternatives
        alternatives = sorted(scores, key=scores.get, reverse=True)[1:4]

        return Recommendation(
            recommended_method=best_method,
            reasoning=reasoning or ["Best overall composite score"],
            tradeoffs=tradeoffs or ["No significant tradeoffs identified"],
            confidence=confidence,
            performance_gain=perf_gain,
            full_report="",
            comparison_data=analysis,
            alternative_methods=alternatives,
        )

    def _build_llm_prompt(
        self,
        results: list[ExperimentResult],
        dataset: DatasetProfile,
        user_context: dict[str, Any],
        analysis: dict[str, Any],
    ) -> str:
        """Build the prompt for LLM recommendation."""
        results_table = self._format_results_table(results)
        context_str = self._format_user_context(user_context)

        prompt = f"""Analyze these ML experiment results and recommend the best method.

**Results:**
{results_table}

**Dataset Characteristics:**
- Task: {dataset.task_type}
- Samples: {dataset.num_samples:,}
- Classes: {dataset.num_classes or 'N/A'}
- Avg sequence length: {dataset.avg_sequence_length:.0f} tokens
- Sequence variance: {dataset.sequence_length_variance}
- Label balance: {dataset.label_balance}
- Domain: {dataset.domain}

**User Context:**
{context_str}

**Pre-computed Scores (for reference):**
{json.dumps(analysis['scores'], indent=2)}

**Your Task:**
1. Identify the best method for this specific use case
2. Explain WHY it's best (2-3 specific, concrete reasons)
3. List tradeoffs or warnings (if any)
4. Estimate confidence in this recommendation

**Response Format (JSON only, no markdown):**
{{
  "recommended_method": "exact method name from results",
  "reasoning": ["specific reason 1", "specific reason 2"],
  "tradeoffs": ["tradeoff 1", "tradeoff 2"],
  "confidence": 0.0-1.0,
  "performance_gain_pct": X.X,
  "alternatives": ["method2", "method3"]
}}"""

        return prompt

    def _format_results_table(self, results: list[ExperimentResult]) -> str:
        """Format results as a readable table."""
        try:
            import pandas as pd

            data = [r.to_dict() for r in results]
            df = pd.DataFrame(data)
            cols = ["method", "accuracy", "f1", "training_time", "memory_peak", "parameters"]
            cols = [c for c in cols if c in df.columns]
            return df[cols].to_string(index=False)
        except ImportError:
            # Fallback without pandas
            lines = ["Method | Accuracy | F1 | Time(h) | Memory(GB) | Params(M)"]
            lines.append("-" * 60)
            for r in results:
                lines.append(
                    f"{r.method} | {r.accuracy:.3f} | {r.f1:.3f} | "
                    f"{r.training_time:.1f} | {r.memory_peak:.1f} | {r.parameters:.1f}"
                )
            return "\n".join(lines)

    def _format_user_context(self, context: dict[str, Any]) -> str:
        """Format user context as readable text."""
        if not context:
            return "No specific constraints provided."

        lines = []
        if "max_training_time" in context:
            lines.append(f"- Max training time: {context['max_training_time']} hours")
        if "max_memory" in context:
            lines.append(f"- Max memory: {context['max_memory']} GB")
        if context.get("prioritize_speed"):
            lines.append("- Priority: Training speed")
        if context.get("prioritize_accuracy"):
            lines.append("- Priority: Maximum accuracy")
        if context.get("memory_constrained"):
            lines.append("- Constraint: Limited GPU memory")

        # Add any other context items
        for key, value in context.items():
            if key not in (
                "max_training_time",
                "max_memory",
                "prioritize_speed",
                "prioritize_accuracy",
                "memory_constrained",
            ):
                lines.append(f"- {key}: {value}")

        return "\n".join(lines) if lines else "No specific constraints provided."

    def _call_llm(self, prompt: str) -> str:
        """Call the LLM with the given prompt."""
        if self.llm is None:
            raise LLMError("No LLM client configured")

        try:
            # PRIORITIZE LangChain interface (what we use with OpenRouter)
            if hasattr(self.llm, "invoke"):
                response = self.llm.invoke(prompt)
                return response.content if hasattr(response, "content") else str(response)
            
            # Then try direct generate
            elif hasattr(self.llm, "generate"):
                return self.llm.generate(prompt)
            
            # Then Anthropic SDK
            elif hasattr(self.llm, "messages"):
                response = self.llm.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return response.content[0].text
            
            # Finally callable
            elif callable(self.llm):
                return self.llm(prompt)
            
            else:
                raise LLMError(f"Unknown LLM client interface: {type(self.llm)}")
        
        except Exception as e:
            raise LLMError(f"LLM call failed: {e}")

    def _parse_llm_response(self, response: str) -> dict[str, Any]:
        """Parse JSON response from LLM."""
        # Try to extract JSON from response
        response = response.strip()

        # Remove markdown code blocks if present
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        # Try to find JSON object
        json_match = re.search(r"\{[\s\S]*\}", response)
        if json_match:
            response = json_match.group()

        try:
            parsed = json.loads(response)
            return parsed
        except json.JSONDecodeError as e:
            raise LLMError(f"Failed to parse LLM response as JSON: {e}")

    def _format_report(
        self,
        recommendation: Recommendation,
        results: list[ExperimentResult],
        dataset: DatasetProfile,
        comparison: dict[str, Any],
    ) -> str:
        """Generate a comprehensive markdown report."""
        best = recommendation.recommended_method
        best_result = next((r for r in results if r.method == best), results[0])

        # Find baseline
        baseline = next(
            (r for r in results if r.method.lower() == self.baseline_method),
            None,
        )

        lines = [
            f"## Recommendation: {best}",
            "",
            "### Why this is best for your use case:",
            "",
        ]

        # Performance section
        if baseline and baseline.method != best:
            acc_gain = (best_result.accuracy - baseline.accuracy) * 100
            lines.append(f"**Performance (+{acc_gain:.1f}% over {baseline.method})**")
            lines.append(
                f"- Accuracy: {best_result.accuracy:.1%} vs "
                f"{baseline.accuracy:.1%} ({baseline.method})"
            )
        else:
            lines.append("**Performance**")
            lines.append(f"- Accuracy: {best_result.accuracy:.1%}")

        lines.append(f"- F1 Score: {best_result.f1:.1%}")
        lines.append("")

        # Add reasoning
        for reason in recommendation.reasoning:
            lines.append(f"- {reason}")
        lines.append("")

        # Efficiency section
        if baseline and baseline.method != best:
            # FIX: Handle division by zero for very fast runs
            if best_result.training_time > 0.0001:
                speedup = baseline.training_time / best_result.training_time
                if speedup > 1:
                    lines.append(f"**Efficiency ({speedup:.1f}x faster training)**")
                else:
                    lines.append("**Efficiency**")
            else:
                 lines.append("**Efficiency (Training time negligible)**")
        else:
            lines.append("**Efficiency**")

        lines.append(f"- Training time: {best_result.training_time:.1f} hours")
        lines.append(f"- Peak memory: {best_result.memory_peak:.1f} GB")
        lines.append(f"- Parameters: {best_result.parameters:.1f}M")
        lines.append("")

        # Tradeoffs section
        if recommendation.tradeoffs:
            lines.append("### Tradeoffs to consider:")
            lines.append("")
            for tradeoff in recommendation.tradeoffs:
                lines.append(f"⚠️  **{tradeoff}**")
            lines.append("")

        # Comparison table
        lines.append("### Full Comparison:")
        lines.append("")
        lines.append(self._format_results_table(results))
        lines.append("")

        # Alternatives
        if recommendation.alternative_methods:
            lines.append("### Alternative Methods:")
            lines.append("")
            for alt in recommendation.alternative_methods[:3]:
                alt_result = next((r for r in results if r.method == alt), None)
                if alt_result:
                    lines.append(
                        f"- **{alt}**: {alt_result.accuracy:.1%} accuracy, "
                        f"{alt_result.training_time:.1f}h training"
                    )
            lines.append("")

        # Confidence
        lines.append("---")
        confidence_label = (
            "High" if recommendation.confidence > 0.8 else
            "Medium" if recommendation.confidence > 0.6 else "Low"
        )
        lines.append(
            f"*Recommendation confidence: {confidence_label} "
            f"({recommendation.confidence:.0%})*"
        )

        return "\n".join(lines)


# Helper function to convert harness metrics to ExperimentResult
def experiment_result_from_metrics(metrics: Dict, method_name: str) -> ExperimentResult:
    """
    Convert harness metrics to ExperimentResult.
    
    Handles our naming convention (training_time_hours, memory_peak_gb, etc.)
    
    Args:
        metrics: Dictionary from harness.py metrics.json
        method_name: Name of the method
    
    Returns:
        ExperimentResult object
    """
    return ExperimentResult(
        method=method_name,
        accuracy=metrics.get('accuracy', 0.0),
        f1=metrics.get('f1', 0.0),
        precision=metrics.get('precision'),
        recall=metrics.get('recall'),
        training_time=metrics.get('training_time_hours', 0.0),
        memory_peak=metrics.get('memory_peak_gb', 0.0),
        parameters=metrics.get('trainable_parameters_millions', 0.0),
        inference_time=metrics.get('inference_time_ms'),
        status=metrics.get('status', 'success'),
        error=metrics.get('error'),
        metadata=metrics
    )


def create_dataset_profile_from_data(
    data: list[dict[str, Any]],
    task_type: str = "classification",
    text_field: str = "text",
    label_field: str = "label",
    domain: str = "general",
) -> DatasetProfile:
    """
    Create a DatasetProfile by analyzing actual data.

    Args:
        data: List of data samples (dictionaries)
        task_type: Type of ML task
        text_field: Field name containing text data
        label_field: Field name containing labels
        domain: Dataset domain

    Returns:
        DatasetProfile with computed statistics
    """
    if not data:
        raise ValidationError("Empty dataset provided")

    # Calculate sequence lengths (rough token estimate: words * 1.3)
    seq_lengths = []
    for sample in data:
        text = sample.get(text_field, "")
        tokens = len(text.split()) * 1.3
        seq_lengths.append(tokens)

    avg_length = sum(seq_lengths) / len(seq_lengths)
    min_length = min(seq_lengths)
    max_length = max(seq_lengths)

    # Calculate variance category
    if seq_lengths:
        variance = sum((x - avg_length) ** 2 for x in seq_lengths) / len(seq_lengths)
        cv = (variance ** 0.5) / avg_length if avg_length > 0 else 0
        if cv < 0.3:
            seq_variance = "low"
        elif cv < 0.7:
            seq_variance = "medium"
        else:
            seq_variance = "high"
    else:
        seq_variance = "medium"

    # Calculate label balance
    labels = [sample.get(label_field) for sample in data if label_field in sample]
    if labels:
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        num_classes = len(label_counts)
        counts = list(label_counts.values())
        max_count, min_count = max(counts), min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float("inf")

        if imbalance_ratio < 1.5:
            balance = "balanced"
        elif imbalance_ratio < 3:
            balance = "slightly_imbalanced"
        elif imbalance_ratio < 10:
            balance = "imbalanced"
        else:
            balance = "highly_imbalanced"
    else:
        num_classes = None
        balance = "balanced"
        label_counts = None

    # Long sequence ratio
    long_threshold = 400
    long_count = sum(1 for length in seq_lengths if length > long_threshold)
    long_ratio = long_count / len(seq_lengths) if seq_lengths else 0

    return DatasetProfile(
        task_type=task_type,
        num_samples=len(data),
        num_classes=num_classes,
        avg_sequence_length=avg_length,
        sequence_length_variance=seq_variance,
        label_balance=balance,
        domain=domain,
        max_sequence_length=int(max_length),
        min_sequence_length=int(min_length),
        long_sequence_ratio=long_ratio,
        class_distribution=label_counts,
    )