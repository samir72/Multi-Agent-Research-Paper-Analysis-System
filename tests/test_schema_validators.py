"""
Unit tests for Pydantic schema validators.
Tests the field_validator decorators in utils/schemas.py.
"""
import pytest
from datetime import datetime

from utils.schemas import Analysis, ConsensusPoint, Contradiction, SynthesisResult


class TestAnalysisValidators:
    """Tests for Analysis schema validators."""

    def test_citations_with_nested_empty_list(self):
        """Test that nested empty lists in citations are flattened."""
        analysis = Analysis(
            paper_id="test_id",
            methodology="Test methodology",
            key_findings=["Finding 1"],
            conclusions="Test conclusions",
            limitations=["Limit 1"],
            citations=["Citation 1", [], "Citation 2"],  # Nested empty list
            main_contributions=["Contribution 1"],
            confidence_score=0.8
        )

        # Should flatten and remove empty lists
        assert analysis.citations == ["Citation 1", "Citation 2"]

    def test_citations_with_deeply_nested_lists(self):
        """Test deeply nested lists are flattened."""
        analysis = Analysis(
            paper_id="test_id",
            methodology="Test",
            key_findings=[["Nested finding"]],
            conclusions="Test",
            limitations=[[["Triple nested"]]],
            citations=[[["Deep citation"]]],
            main_contributions=[],
            confidence_score=0.5
        )

        assert analysis.key_findings == ["Nested finding"]
        assert analysis.limitations == ["Triple nested"]
        assert analysis.citations == ["Deep citation"]

    def test_mixed_types_are_normalized(self):
        """Test that mixed types in lists are handled."""
        analysis = Analysis(
            paper_id="test_id",
            methodology="Test",
            key_findings=["Finding", None, 123, ""],
            conclusions="Test",
            limitations=[456, "Limit"],
            citations=["Citation", None, ""],
            confidence_score=0.7
        )

        # None and empty strings filtered out, numbers converted to strings
        assert analysis.key_findings == ["Finding", "123"]
        assert analysis.limitations == ["456", "Limit"]
        assert analysis.citations == ["Citation"]

    def test_string_converted_to_list(self):
        """Test that strings in list fields are converted to single-element lists."""
        analysis = Analysis(
            paper_id="test_id",
            methodology="Test",
            key_findings="Single finding",  # String instead of list
            conclusions="Test",
            limitations="Single limitation",  # String instead of list
            citations=[],
            confidence_score=0.6
        )

        assert analysis.key_findings == ["Single finding"]
        assert analysis.limitations == ["Single limitation"]


class TestConsensusPointValidators:
    """Tests for ConsensusPoint schema validators."""

    def test_supporting_papers_with_nested_lists(self):
        """Test that nested lists in supporting_papers are flattened."""
        cp = ConsensusPoint(
            statement="Test consensus",
            supporting_papers=["paper1", [], ["paper2"]],
            citations=["Citation 1", [["Nested citation"]]],
            confidence=0.9
        )

        assert cp.supporting_papers == ["paper1", "paper2"]
        assert cp.citations == ["Citation 1", "Nested citation"]

    def test_empty_and_none_values_filtered(self):
        """Test that None and empty strings are filtered."""
        cp = ConsensusPoint(
            statement="Test",
            supporting_papers=["paper1", None, "", "paper2"],
            citations=["Citation", None],
            confidence=0.8
        )

        assert cp.supporting_papers == ["paper1", "paper2"]
        assert cp.citations == ["Citation"]


class TestContradictionValidators:
    """Tests for Contradiction schema validators."""

    def test_papers_lists_with_nested_values(self):
        """Test that nested lists in papers_a and papers_b are flattened."""
        contr = Contradiction(
            topic="Test topic",
            viewpoint_a="View A",
            papers_a=["paper1", [], "paper2"],
            viewpoint_b="View B",
            papers_b=[["paper3"], "paper4"],
            citations=["Citation 1", [["Nested"]]],
            confidence=0.7
        )

        assert contr.papers_a == ["paper1", "paper2"]
        assert contr.papers_b == ["paper3", "paper4"]
        assert contr.citations == ["Citation 1", "Nested"]

    def test_mixed_types_normalized(self):
        """Test mixed types in papers lists."""
        contr = Contradiction(
            topic="Test",
            viewpoint_a="A",
            papers_a=["paper1", 123, None],
            viewpoint_b="B",
            papers_b=[456, "paper2"],
            citations=["Citation"],
            confidence=0.6
        )

        assert contr.papers_a == ["paper1", "123"]
        assert contr.papers_b == ["456", "paper2"]


class TestSynthesisResultValidators:
    """Tests for SynthesisResult schema validators."""

    def test_research_gaps_with_nested_lists(self):
        """Test that nested lists in research_gaps are flattened."""
        synthesis = SynthesisResult(
            consensus_points=[],
            contradictions=[],
            research_gaps=["Gap 1", [["Nested gap"]], None],
            summary="Test summary",
            confidence_score=0.8,
            papers_analyzed=["paper1", [], "paper2"]
        )

        assert synthesis.research_gaps == ["Gap 1", "Nested gap"]
        assert synthesis.papers_analyzed == ["paper1", "paper2"]

    def test_string_converted_to_list(self):
        """Test that strings are converted to lists."""
        synthesis = SynthesisResult(
            consensus_points=[],
            contradictions=[],
            research_gaps="Single gap",  # String instead of list
            summary="Test",
            confidence_score=0.7,
            papers_analyzed="paper1"  # String instead of list
        )

        assert synthesis.research_gaps == ["Single gap"]
        assert synthesis.papers_analyzed == ["paper1"]


class TestValidatorsWithRealWorldData:
    """Tests simulating real-world LLM response edge cases."""

    def test_llm_returns_empty_arrays_within_citations(self):
        """Simulate the exact bug reported: citations contains empty lists."""
        # This is the bug: ["citation 1", [], "citation 2"]
        analysis = Analysis(
            paper_id="2303.08710v1",
            methodology="Deep learning approach",
            key_findings=["95% accuracy", [], "Outperforms baselines"],
            conclusions="Novel method works well",
            limitations=["Limited dataset", []],
            citations=["Methodology section", [], "Results section"],
            main_contributions=["Novel architecture"],
            confidence_score=0.85
        )

        # Should successfully create Analysis without Pydantic validation errors
        assert isinstance(analysis, Analysis)
        assert analysis.citations == ["Methodology section", "Results section"]
        assert analysis.key_findings == ["95% accuracy", "Outperforms baselines"]
        assert analysis.limitations == ["Limited dataset"]

    def test_llm_returns_mixed_malformed_data(self):
        """Test extremely malformed data that might come from LLM."""
        analysis = Analysis(
            paper_id="test_id",
            methodology="Test",
            key_findings=[[], "Finding", None, [["Nested"]], "", "  ", 123],
            conclusions="Test",
            limitations=[[["Deep"]], None, "Limit", []],
            citations=["Citation", [[], []], None, ""],
            main_contributions=[None, [], "Contribution", [["Deep contrib"]]],
            confidence_score=0.5
        )

        # All malformed data should be cleaned
        assert analysis.key_findings == ["Finding", "Nested", "123"]
        assert analysis.limitations == ["Deep", "Limit"]
        assert analysis.citations == ["Citation"]
        assert analysis.main_contributions == ["Contribution", "Deep contrib"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
