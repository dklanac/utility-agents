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

"""Bill Investigator Agent Package.

This package provides an AI agent for analyzing utility usage patterns
and investigating unexpected bill increases.
"""

from .core import root_agent
from .output_formatting import (
    Hypothesis,
    Evidence,
    AnalysisResult,
    HypothesisCategory,
    ConfidenceLevel,
    rank_hypotheses,
    format_conversational_output,
    save_analysis_to_session,
    get_analysis_from_session,
    get_analysis_history,
    clear_analysis_session,
)

__all__ = [
    "root_agent",
    "Hypothesis",
    "Evidence",
    "AnalysisResult",
    "HypothesisCategory",
    "ConfidenceLevel",
    "rank_hypotheses",
    "format_conversational_output",
    "save_analysis_to_session",
    "get_analysis_from_session",
    "get_analysis_history",
    "clear_analysis_session",
]
