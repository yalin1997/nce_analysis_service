from analysis_core.stats import holm_bonferroni_adjust


def test_empty_input_returns_empty():
    assert holm_bonferroni_adjust([]) == []


def test_single_p_value_unchanged():
    assert holm_bonferroni_adjust([0.03]) == [0.03]


def test_adjusts_in_rank_order_with_running_max():
    # sorted: 0.01 -> 3*0.01=0.03; 0.03 -> 2*0.03=0.06; 0.04 -> max(1*0.04, 0.06)=0.06
    assert holm_bonferroni_adjust([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]


def test_caps_at_one():
    assert holm_bonferroni_adjust([0.9, 0.8]) == [1.0, 1.0]
