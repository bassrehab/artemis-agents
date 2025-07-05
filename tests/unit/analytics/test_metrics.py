"""Tests for metrics calculation in ARTEMIS analytics."""

from unittest.mock import MagicMock

import pytest

from artemis.analytics.metrics import DebateMetricsCalculator, RebuttalAnalyzer


def create_mock_turn(
    agent: str,
    round_num: int,
    score: float,
    level: str = "strategic",
    evidence_count: int = 0,
    has_rebuttal: bool = False,
    rebutted_ids: list[str] | None = None,
):
    """Create a mock Turn object for testing."""
    turn = MagicMock()
    turn.id = f"turn_{agent}_{round_num}"
    turn.agent = agent
    turn.round = round_num
    turn.evaluation = MagicMock()
    turn.evaluation.total_score = score
    turn.argument = MagicMock()
    turn.argument.rebuts = rebutted_ids or (["prev_turn"] if has_rebuttal else [])
    turn.argument.level = MagicMock()
    turn.argument.level.value = level

    # Create mock evidence
    turn.argument.evidence = []
    for i in range(evidence_count):
        ev = MagicMock()
        ev.confidence = 0.8
        ev.verified = i % 2 == 0  # Alternate verified
        ev.source = f"Source {i}"
        turn.argument.evidence.append(ev)

    # Create mock causal links
    turn.argument.causal_links = []
    if level == "tactical":
        link = MagicMock()
        link.cause = "economic growth"
        link.effect = "employment rate"
        turn.argument.causal_links.append(link)

    return turn


class TestDebateMetricsCalculator:
    """Tests for DebateMetricsCalculator class."""

    def test_init(self):
        """Test initialization."""
        transcript = []
        agents = ["pro", "con"]
        calc = DebateMetricsCalculator(transcript, agents)
        assert calc._agents == agents

    def test_compute_all_empty_transcript(self):
        """Test computing metrics from empty transcript."""
        calc = DebateMetricsCalculator([], ["pro", "con"])
        metrics = calc.compute_all()

        assert "rebuttal_effectiveness" in metrics
        assert "evidence_utilization" in metrics
        assert "argument_diversity_index" in metrics

    def test_rebuttal_effectiveness(self):
        """Test rebuttal effectiveness calculation."""
        transcript = [
            create_mock_turn("pro", 1, 0.7, has_rebuttal=False),
            create_mock_turn("con", 1, 0.6, has_rebuttal=True),  # Rebuttal
            create_mock_turn("pro", 2, 0.8, has_rebuttal=True),  # Strong rebuttal
        ]

        calc = DebateMetricsCalculator(transcript, ["pro", "con"])
        effectiveness = calc.rebuttal_effectiveness

        assert "pro" in effectiveness
        assert "con" in effectiveness
        assert 0 <= effectiveness["pro"] <= 1
        assert 0 <= effectiveness["con"] <= 1

    def test_evidence_utilization(self):
        """Test evidence utilization calculation."""
        transcript = [
            create_mock_turn("pro", 1, 0.7, evidence_count=3),
            create_mock_turn("con", 1, 0.6, evidence_count=1),
            create_mock_turn("pro", 2, 0.8, evidence_count=2),
            create_mock_turn("con", 2, 0.5, evidence_count=0),
        ]

        calc = DebateMetricsCalculator(transcript, ["pro", "con"])
        utilization = calc.evidence_utilization

        assert "pro" in utilization
        assert "con" in utilization
        # Pro should have higher utilization (more evidence)
        assert utilization["pro"] > utilization["con"]

    def test_argument_diversity_index(self):
        """Test argument diversity index calculation."""
        transcript = [
            # Pro uses all levels - high diversity
            create_mock_turn("pro", 1, 0.7, level="strategic"),
            create_mock_turn("pro", 2, 0.7, level="tactical"),
            create_mock_turn("pro", 3, 0.7, level="operational"),
            # Con only uses strategic - low diversity
            create_mock_turn("con", 1, 0.6, level="strategic"),
            create_mock_turn("con", 2, 0.6, level="strategic"),
            create_mock_turn("con", 3, 0.6, level="strategic"),
        ]

        calc = DebateMetricsCalculator(transcript, ["pro", "con"])
        diversity = calc.argument_diversity_index

        assert "pro" in diversity
        assert "con" in diversity
        # Pro should have higher diversity (uses all levels)
        assert diversity["pro"] > diversity["con"]

    def test_topic_coverage(self):
        """Test topic coverage extraction."""
        transcript = [
            create_mock_turn("pro", 1, 0.7, level="tactical"),
            create_mock_turn("con", 1, 0.6, level="tactical"),
        ]

        calc = DebateMetricsCalculator(transcript, ["pro", "con"])
        coverage = calc.topic_coverage

        assert "pro" in coverage
        assert "con" in coverage
        assert isinstance(coverage["pro"], list)

    def test_get_round_metrics(self):
        """Test getting metrics for a specific round."""
        transcript = [
            create_mock_turn("pro", 1, 0.7),
            create_mock_turn("con", 1, 0.6),
            create_mock_turn("pro", 2, 0.8),
            create_mock_turn("con", 2, 0.5),
        ]

        calc = DebateMetricsCalculator(transcript, ["pro", "con"])
        round1_metrics = calc.get_round_metrics(1)

        assert round1_metrics.round == 1
        assert "pro" in round1_metrics.agent_scores
        assert "con" in round1_metrics.agent_scores
        assert round1_metrics.agent_scores["pro"] == 0.7
        assert round1_metrics.agent_scores["con"] == 0.6

    def test_get_all_round_metrics(self):
        """Test getting metrics for all rounds."""
        transcript = [
            create_mock_turn("pro", 1, 0.7),
            create_mock_turn("con", 1, 0.6),
            create_mock_turn("pro", 2, 0.8),
            create_mock_turn("con", 2, 0.5),
        ]

        calc = DebateMetricsCalculator(transcript, ["pro", "con"])
        all_metrics = calc.get_all_round_metrics()

        assert len(all_metrics) == 2
        assert all_metrics[0].round == 1
        assert all_metrics[1].round == 2

    def test_score_delta_calculation(self):
        """Test score delta between rounds."""
        transcript = [
            create_mock_turn("pro", 1, 0.6),
            create_mock_turn("con", 1, 0.6),
            create_mock_turn("pro", 2, 0.8),  # Improved
            create_mock_turn("con", 2, 0.4),  # Declined
        ]

        calc = DebateMetricsCalculator(transcript, ["pro", "con"])
        round2_metrics = calc.get_round_metrics(2)

        assert round2_metrics.score_delta["pro"] > 0  # Pro improved
        assert round2_metrics.score_delta["con"] < 0  # Con declined


class TestRebuttalAnalyzer:
    """Tests for RebuttalAnalyzer class."""

    def test_analyze_rebuttal_chain(self):
        """Test rebuttal chain analysis."""
        turn1 = create_mock_turn("pro", 1, 0.7)
        turn2 = create_mock_turn("con", 1, 0.6, rebutted_ids=[turn1.id])

        analyzer = RebuttalAnalyzer([turn1, turn2])
        chains = analyzer.analyze_rebuttal_chain()

        assert len(chains) == 1
        assert chains[0]["argument_id"] == turn1.id
        assert chains[0]["rebutted_by"] == turn2.id

    def test_compute_rebuttal_success_rate(self):
        """Test rebuttal success rate calculation."""
        turn1 = create_mock_turn("pro", 1, 0.6)
        turn2 = create_mock_turn("con", 1, 0.8, rebutted_ids=[turn1.id])  # Successful
        turn3 = create_mock_turn("pro", 2, 0.7)
        turn4 = create_mock_turn("con", 2, 0.5, rebutted_ids=[turn3.id])  # Failed

        analyzer = RebuttalAnalyzer([turn1, turn2, turn3, turn4])
        success_rate = analyzer.compute_rebuttal_success_rate("con")

        # One successful (0.8 > 0.6), one failed (0.5 < 0.7)
        assert success_rate == 0.5

    def test_get_most_rebutted_arguments(self):
        """Test finding most rebutted arguments."""
        turn1 = create_mock_turn("pro", 1, 0.7)
        turn2 = create_mock_turn("con", 1, 0.6, rebutted_ids=[turn1.id])
        turn3 = create_mock_turn("con", 2, 0.7, rebutted_ids=[turn1.id])  # Same target

        analyzer = RebuttalAnalyzer([turn1, turn2, turn3])
        most_rebutted = analyzer.get_most_rebutted_arguments(top_n=5)

        assert len(most_rebutted) == 1
        assert most_rebutted[0]["argument_id"] == turn1.id
        assert most_rebutted[0]["rebuttal_count"] == 2
