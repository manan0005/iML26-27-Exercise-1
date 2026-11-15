"""Run the interpretability evaluation and store the results as a CSV file."""

from __future__ import annotations

from pathlib import Path
import ssl

# --- Quick fix for macOS SSL certificate download error ---
ssl._create_default_https_context = ssl._create_unverified_context

from iml_interpretability.metric import InterpretabilityComplexityScore
from iml_interpretability.evaluate import evaluate
from iml_interpretability.models import get_models
from iml_interpretability.data import load_titanic

OUTPUT = Path("results/scores.csv")
SEED = 0

def main() -> None:
    results = evaluate(
        datasets=[load_titanic()],
        models=get_models(seed=SEED),
        metrics=[InterpretabilityComplexityScore()],
        seed=SEED,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(OUTPUT, index=False)
    print(results.to_string(index=False))

if __name__ == "__main__":
    main()