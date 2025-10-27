# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Output formatting and session management for Bill Investigator.

This module provides structured output formatting with confidence scoring,
hypothesis ranking, and dual-format output (JSON + conversational text).
"""

import logging
import time
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

from google.adk.agents.callback_context import CallbackContext

# Configure logging
logger = logging.getLogger(__name__)


class HypothesisCategory(Enum):
    """Categories for usage anomaly hypotheses."""
    HVAC_INCREASE = "hvac_increase"
    TIME_SHIFT = "time_shift"
    NEW_LOAD = "new_load"
    BASELINE_INCREASE = "baseline_increase"
    EQUIPMENT_ISSUE = "equipment_issue"
    BEHAVIORAL_CHANGE = "behavioral_change"
    SEASONAL_WEATHER = "seasonal_weather"


class ConfidenceLevel(Enum):
    """Confidence level classifications."""
    VERY_HIGH = "very_high"  # 0.8-1.0
    HIGH = "high"            # 0.6-0.8
    MEDIUM = "medium"        # 0.4-0.6
    LOW = "low"              # 0.2-0.4
    VERY_LOW = "very_low"    # 0.0-0.2


@dataclass
class Evidence:
    """Individual piece of evidence supporting a hypothesis.

    Attributes:
        description: Natural language description of the evidence
        data_source: Where this evidence came from (e.g., "hourly_average query")
        metric_name: The metric being referenced (e.g., "hvac_cooling_kwh")
        value: The actual value or change observed
        comparison: Optional comparison context (e.g., "vs. last month: 45.2 kWh")
    """
    description: str
    data_source: str
    metric_name: str
    value: Any
    comparison: Optional[str] = None


@dataclass
class Hypothesis:
    """A hypothesis about the cause of usage changes.

    Attributes:
        description: Natural language description of the hypothesis
        category: Category of this hypothesis
        confidence_score: Confidence score from 0.0 to 1.0
        evidence_items: List of evidence supporting this hypothesis
        priority_rank: Rank among all hypotheses (1 = highest priority)
        recommendations: Suggested actions to verify or address this hypothesis
        potential_savings: Estimated potential savings if addressed (in dollars)
    """
    description: str
    category: HypothesisCategory
    confidence_score: float
    evidence_items: List[Evidence] = field(default_factory=list)
    priority_rank: int = 0
    recommendations: List[str] = field(default_factory=list)
    potential_savings: Optional[float] = None

    def __post_init__(self):
        """Validate confidence score."""
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError(f"Confidence score must be between 0 and 1, got {self.confidence_score}")

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level classification."""
        if self.confidence_score >= 0.8:
            return ConfidenceLevel.VERY_HIGH
        elif self.confidence_score >= 0.6:
            return ConfidenceLevel.HIGH
        elif self.confidence_score >= 0.4:
            return ConfidenceLevel.MEDIUM
        elif self.confidence_score >= 0.2:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW


@dataclass
class AnalysisMetadata:
    """Metadata about the analysis process.

    Attributes:
        timestamp: When the analysis was performed
        customer_id: Customer identifier
        analysis_period: Date range analyzed
        baseline_period: Comparison baseline period
        analysis_duration_ms: How long the analysis took
        data_sources_used: List of data sources queried
        total_hypotheses: Number of hypotheses generated
    """
    timestamp: str
    customer_id: str
    analysis_period: str
    baseline_period: Optional[str] = None
    analysis_duration_ms: Optional[float] = None
    data_sources_used: List[str] = field(default_factory=list)
    total_hypotheses: int = 0


@dataclass
class AnalysisResult:
    """Complete analysis result with ranked hypotheses.

    Attributes:
        metadata: Analysis metadata
        hypotheses: Ranked list of hypotheses
        summary: Brief summary of findings
        key_findings: List of key findings
    """
    metadata: AnalysisMetadata
    hypotheses: List[Hypothesis] = field(default_factory=list)
    summary: str = ""
    key_findings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "metadata": asdict(self.metadata),
            "hypotheses": [
                {
                    **asdict(h),
                    "category": h.category.value,
                    "confidence_level": h.confidence_level.value,
                    "evidence_items": [asdict(e) for e in h.evidence_items]
                }
                for h in self.hypotheses
            ],
            "summary": self.summary,
            "key_findings": self.key_findings
        }


def calculate_hypothesis_score(
    hypothesis: Hypothesis,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Calculate weighted score for hypothesis ranking.

    Args:
        hypothesis: The hypothesis to score
        weights: Optional custom weights for scoring factors

    Returns:
        Weighted score for ranking
    """
    if weights is None:
        weights = {
            "confidence": 0.5,      # Primary factor
            "evidence_count": 0.3,  # More evidence = higher rank
            "has_savings": 0.2,     # Actionable hypotheses ranked higher
        }

    # Base score from confidence
    score = hypothesis.confidence_score * weights["confidence"]

    # Evidence count factor (normalized to 0-1, max at 5 pieces of evidence)
    evidence_factor = min(len(hypothesis.evidence_items) / 5.0, 1.0)
    score += evidence_factor * weights["evidence_count"]

    # Actionability factor (has potential savings estimate)
    if hypothesis.potential_savings is not None and hypothesis.potential_savings > 0:
        score += weights["has_savings"]

    return score


def rank_hypotheses(
    hypotheses: List[Hypothesis],
    weights: Optional[Dict[str, float]] = None
) -> List[Hypothesis]:
    """Rank hypotheses by calculated score.

    Args:
        hypotheses: List of hypotheses to rank
        weights: Optional custom weights for scoring

    Returns:
        Sorted list with priority_rank assigned
    """
    start_time = time.time()

    # Calculate scores
    scored = [
        (h, calculate_hypothesis_score(h, weights))
        for h in hypotheses
    ]

    # Sort by score (descending)
    scored.sort(key=lambda x: x[1], reverse=True)

    # Assign priority ranks
    ranked = []
    for rank, (hypothesis, score) in enumerate(scored, start=1):
        hypothesis.priority_rank = rank
        ranked.append(hypothesis)
        logger.debug(
            f"Ranked hypothesis #{rank}: {hypothesis.description[:50]}... "
            f"(score={score:.3f}, confidence={hypothesis.confidence_score:.2f})"
        )

    duration = (time.time() - start_time) * 1000
    logger.info(f"Ranked {len(hypotheses)} hypotheses in {duration:.2f}ms")

    return ranked


def format_confidence_text(confidence_score: float) -> str:
    """Convert confidence score to natural language.

    Args:
        confidence_score: Score from 0.0 to 1.0

    Returns:
        Natural language confidence descriptor
    """
    if confidence_score >= 0.8:
        return "very likely"
    elif confidence_score >= 0.6:
        return "likely"
    elif confidence_score >= 0.4:
        return "possible"
    elif confidence_score >= 0.2:
        return "unlikely but possible"
    else:
        return "low probability"


def format_conversational_output(result: AnalysisResult) -> str:
    """Format analysis result as conversational text.

    Args:
        result: The analysis result to format

    Returns:
        Markdown-formatted conversational text
    """
    lines = []

    # Header
    lines.append("# Utility Bill Analysis Report")
    lines.append(f"\n**Customer:** {result.metadata.customer_id}")
    lines.append(f"**Analysis Period:** {result.metadata.analysis_period}")
    if result.metadata.baseline_period:
        lines.append(f"**Baseline Period:** {result.metadata.baseline_period}")
    lines.append(f"**Date:** {result.metadata.timestamp}\n")

    # Summary
    lines.append("## Summary\n")
    lines.append(result.summary + "\n")

    # Key Findings
    if result.key_findings:
        lines.append("## Key Findings\n")
        for finding in result.key_findings:
            lines.append(f"- {finding}")
        lines.append("")

    # Hypotheses
    lines.append("## Likely Causes (Ranked by Confidence)\n")

    for hypothesis in result.hypotheses:
        confidence_text = format_confidence_text(hypothesis.confidence_score)

        lines.append(f"### {hypothesis.priority_rank}. {hypothesis.description}")
        lines.append(f"**Confidence:** {confidence_text.title()} ({hypothesis.confidence_score:.0%})\n")

        # Evidence
        if hypothesis.evidence_items:
            lines.append("**Supporting Evidence:**")
            for evidence in hypothesis.evidence_items:
                evidence_line = f"- {evidence.description}"
                if evidence.comparison:
                    evidence_line += f" ({evidence.comparison})"
                lines.append(evidence_line)
            lines.append("")

        # Recommendations
        if hypothesis.recommendations:
            lines.append("**Recommended Actions:**")
            for rec in hypothesis.recommendations:
                lines.append(f"- {rec}")
            lines.append("")

        # Potential savings
        if hypothesis.potential_savings:
            lines.append(f"**Potential Monthly Savings:** ${hypothesis.potential_savings:.2f}\n")

    # Footer
    lines.append("---")
    lines.append(f"\n*Analysis completed in {result.metadata.analysis_duration_ms:.0f}ms*")

    return "\n".join(lines)


def create_analysis_result(
    customer_id: str,
    analysis_period: str,
    baseline_period: Optional[str] = None,
    start_time: Optional[float] = None
) -> AnalysisResult:
    """Create a new analysis result with metadata.

    Args:
        customer_id: Customer identifier
        analysis_period: Date range being analyzed
        baseline_period: Optional baseline comparison period
        start_time: Optional start time for duration calculation

    Returns:
        New AnalysisResult with initialized metadata
    """
    metadata = AnalysisMetadata(
        timestamp=datetime.now().isoformat(),
        customer_id=customer_id,
        analysis_period=analysis_period,
        baseline_period=baseline_period,
        analysis_duration_ms=None
    )

    if start_time:
        metadata.analysis_duration_ms = (time.time() - start_time) * 1000

    return AnalysisResult(metadata=metadata)


def save_analysis_to_session(
    callback_context: CallbackContext,
    result: AnalysisResult,
    output_key: str = "analysis_result"
) -> None:
    """Save analysis result to session state.

    Args:
        callback_context: The callback context containing session state
        result: The analysis result to save
        output_key: Key to use for storing in session state
    """
    if not hasattr(callback_context, 'state'):
        logger.warning("No session state available, skipping save")
        return

    # Save the full result
    callback_context.state[output_key] = result.to_dict()

    # Also save to analysis history
    if "analysis_history" not in callback_context.state:
        callback_context.state["analysis_history"] = []

    callback_context.state["analysis_history"].append({
        "timestamp": result.metadata.timestamp,
        "customer_id": result.metadata.customer_id,
        "analysis_period": result.metadata.analysis_period,
        "num_hypotheses": len(result.hypotheses),
        "top_hypothesis": result.hypotheses[0].description if result.hypotheses else None
    })

    logger.info(
        f"Saved analysis result to session: {result.metadata.customer_id} "
        f"({len(result.hypotheses)} hypotheses)"
    )


def get_analysis_from_session(
    callback_context: CallbackContext,
    output_key: str = "analysis_result"
) -> Optional[Dict]:
    """Retrieve analysis result from session state.

    Args:
        callback_context: The callback context containing session state
        output_key: Key to retrieve from session state

    Returns:
        Analysis result dictionary or None if not found
    """
    if not hasattr(callback_context, 'state'):
        logger.warning("No session state available")
        return None

    result = callback_context.state.get(output_key)

    if result:
        logger.info(f"Retrieved analysis result from session for {result['metadata']['customer_id']}")
    else:
        logger.debug(f"No analysis result found in session with key '{output_key}'")

    return result


def get_analysis_history(
    callback_context: CallbackContext
) -> List[Dict]:
    """Get list of previous analyses from session.

    Args:
        callback_context: The callback context containing session state

    Returns:
        List of analysis history entries
    """
    if not hasattr(callback_context, 'state'):
        return []

    return callback_context.state.get("analysis_history", [])


def clear_analysis_session(
    callback_context: CallbackContext,
    clear_history: bool = False
) -> None:
    """Clear current analysis from session.

    Args:
        callback_context: The callback context containing session state
        clear_history: Whether to also clear analysis history
    """
    if not hasattr(callback_context, 'state'):
        return

    if "analysis_result" in callback_context.state:
        del callback_context.state["analysis_result"]
        logger.info("Cleared current analysis from session")

    if clear_history and "analysis_history" in callback_context.state:
        del callback_context.state["analysis_history"]
        logger.info("Cleared analysis history from session")


# Example usage and testing
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    print("=" * 70)
    print("Output Formatting Module Test")
    print("=" * 70)
    print()

    # Create sample hypotheses
    h1 = Hypothesis(
        description="HVAC cooling load increased by 40% due to thermostat setting change",
        category=HypothesisCategory.HVAC_INCREASE,
        confidence_score=0.85,
        evidence_items=[
            Evidence(
                description="HVAC cooling usage increased from 450 kWh to 630 kWh",
                data_source="appliance_breakdown query",
                metric_name="hvac_cooling_kwh",
                value=630,
                comparison="vs. last month: 450 kWh"
            ),
            Evidence(
                description="Cooling degree-hours only increased by 10%",
                data_source="seasonal_trends query",
                metric_name="cooling_degree_hours",
                value=1650,
                comparison="vs. last month: 1500"
            ),
        ],
        recommendations=[
            "Check thermostat settings - recommend 78°F for cooling",
            "Inspect HVAC filter and clean if needed",
            "Consider programmable thermostat for automated temperature control"
        ],
        potential_savings=45.0
    )

    h2 = Hypothesis(
        description="New EV charging pattern started mid-month",
        category=HypothesisCategory.NEW_LOAD,
        confidence_score=0.72,
        evidence_items=[
            Evidence(
                description="EV charging usage appeared starting July 15",
                data_source="hourly_average query",
                metric_name="ev_charging_kwh",
                value=120,
                comparison="Previous months: 0 kWh"
            ),
        ],
        recommendations=[
            "Verify new EV purchase or charging routine",
            "Consider off-peak charging (midnight-6am) for lower rates"
        ],
        potential_savings=None
    )

    h3 = Hypothesis(
        description="Pool pump runtime extended for summer season",
        category=HypothesisCategory.SEASONAL_WEATHER,
        confidence_score=0.55,
        evidence_items=[
            Evidence(
                description="Pool equipment usage increased by 25%",
                data_source="appliance_breakdown query",
                metric_name="pool_equipment_kwh",
                value=85,
                comparison="vs. last month: 68 kWh"
            ),
        ],
        recommendations=[
            "Verify pool pump timer settings",
            "Consider reducing runtime to 6 hours/day if water quality allows"
        ],
        potential_savings=12.0
    )

    # Create analysis result
    result = create_analysis_result(
        customer_id="CUST_003",
        analysis_period="2024-07-01 to 2024-07-31",
        baseline_period="2024-06-01 to 2024-06-30",
        start_time=time.time() - 2.5  # Simulate 2.5 second analysis
    )

    result.summary = "Usage increased by 285 kWh (28%) compared to last month. Primary driver appears to be increased HVAC cooling, with secondary contribution from new EV charging."

    result.key_findings = [
        "HVAC cooling usage up 180 kWh (40% increase)",
        "New EV charging pattern started mid-month (~120 kWh)",
        "Pool equipment usage up 17 kWh (25% increase)",
        "Overall usage: 1,305 kWh vs. 1,020 kWh baseline"
    ]

    # Add hypotheses
    result.hypotheses = [h1, h2, h3]

    # Rank hypotheses
    result.hypotheses = rank_hypotheses(result.hypotheses)
    result.metadata.total_hypotheses = len(result.hypotheses)

    # Test JSON output
    print("JSON Output:")
    print("-" * 70)
    import json
    print(json.dumps(result.to_dict(), indent=2))
    print()

    # Test conversational output
    print("\nConversational Output:")
    print("-" * 70)
    print(format_conversational_output(result))
    print()

    print("=" * 70)
    print("✓ Output formatting module tested successfully!")
    print("=" * 70)
