import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
GENERATED = ROOT / "paper/generated"
PAPER = (ROOT / "paper/main.tex").read_text(encoding="utf-8")


def _macros() -> dict[str, str]:
    text = (GENERATED / "audit_metrics.tex").read_text(encoding="utf-8")
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", text))


def test_regime_macros_match_corrected_labels() -> None:
    macros = _macros()
    labels = pd.read_csv(ROOT / "results/processed/regime_labels_v2.csv")
    counts = labels.oracle_label.value_counts()
    assert macros["VTwoPracticalTies"] == str(counts["practical_tie"])
    assert macros["VTwoRawWinners"] == str(counts["raw"])
    assert macros["VTwoReadoutWinners"] == str(counts["readout"])
    assert macros["VTwoZNEWinners"] == str(counts["zne"])


def test_audit_macros_match_the_committed_audit() -> None:
    macros = _macros()
    audit = pd.read_csv(ROOT / "results/processed/selector_audit_v1.csv")
    selector = audit.loc[audit.policy == "selector"]
    assert macros["SelectorCatastrophic"] == str(
        int((selector.audit_category == "catastrophic_regret").sum())
    )
    assert macros["SelectorAbstentions"] == str(int((selector.selected_method == "abstain").sum()))


def test_paper_uses_corrected_results_and_defines_every_macro() -> None:
    assert r"\PracticalTies{}" not in PAPER
    assert "unseen synthetic noise" not in PAPER
    assert "outcomes are not reported here" in PAPER
    assert r"\input{generated/audit_metrics.tex}" in PAPER
    macros = _macros()
    used = set(re.findall(r"\\(VTwo\w+|Selector[A-Z]\w+|Null\w+|NearTie\w+|Biased\w+)\{\}", PAPER))
    assert used <= set(macros)


def test_paper_tables_and_figures_exist() -> None:
    for name in re.findall(r"\\input\{(generated/[^}]+)\}", PAPER):
        assert (ROOT / "paper" / name).exists()
    for name in re.findall(r"\\includegraphics\[[^]]*\]\{([^}]+)\}", PAPER):
        assert (ROOT / "paper/figures" / name).exists()
