# iML 26/27 – Exercise 1: Quantifying Interpretability

## My Approach: Interpretability Complexity Score (ICS)
To quantify model interpretability, I developed the **Interpretability Complexity Score (ICS)**. For inherently interpretable models like Decision Trees, interpretability is inversely proportional to the model's structural complexity and the cognitive load required for a human to comprehend and simulate it.

The ICS is calculated using the following formula:
`ICS = (alpha * Total Nodes) + (beta * Maximum Depth) + (gamma * Unique Features Used)`
*(Where alpha=1.0, beta=2.0, and gamma=5.0)*

For ensembles (like Random Forests), the score is the sum of the ICS of all individual trees in the ensemble. A **lower ICS** indicates a **more interpretable** model. 

### Task 1 Findings: The Threshold of Interpretability
I evaluated this metric on the Titanic dataset across single Decision Trees of varying depths and Random Forests of varying sizes. 

My results were as follows:
- **Tree (Depth 2)**: Accuracy 77.1%, ICS 26.0
- **Tree (Depth 5)**: Accuracy 79.8%, ICS 91.0
- **Tree (Unrestricted)**: Accuracy 75.3%, ICS 535.0
- **Forest (3 Trees)**: Accuracy 81.0%, ICS 133.0
- **Forest (10 Trees)**: Accuracy 81.4%, ICS 1002.0
- **Forest (50 Trees)**: Accuracy 77.7%, ICS 25552.0

**Conclusion on the Threshold:** The "threshold of interpretability" is clearly crossed when moving from bounded trees to ensembles. While a human can mentally trace a depth-5 tree (ICS 91.0), a 10-tree forest (ICS 1002.0) crosses the threshold of human working memory entirely, despite only providing a marginal ~1.5% gain in accuracy over the single depth-5 tree. By 50 trees, the ICS explodes to over 25,000, completely eliminating interpretability.

### Task 2 Findings: Meta-Features and Interpretability
To understand how dataset meta-features impact model interpretability, I trained unrestricted Decision Trees on three datasets with varying complexity:
- **Iris:** 150 Samples, 4 Features, 33.3% Majority Class ➔ ICS: 46.0
- **Wine:** 178 Samples, 13 Features, 39.9% Majority Class ➔ ICS: 46.0
- **Breast Cancer:** 569 Samples, 30 Features, 62.7% Majority Class ➔ ICS: 95.0

**Conclusion:** The data suggests that dataset complexity generally degrades model interpretability, though it is heavily confounded by class separability. Interestingly, the model achieved the exact same complexity on the Wine dataset as the Iris dataset, despite having 3x the features. This indicates the tree found highly efficient splits using only a small subset of features. However, a significant jump in samples, features, and class imbalance (Breast Cancer) forced the tree to grow significantly deeper and wider to achieve node purity.


## How to Reproduce
To reproduce these results, set up the virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

Run Task 1 (Titanic evaluation):
```bash
python scripts/run_experiment.py
```

Run Task 2 (Meta-features evaluation):
```bash
python scripts/run_task2.py
```

# AI Statement

I used Gemini as an AI assistant for this exercise. I used it to help brainstorm the cognitive metrics for the ICS formula, troubleshoot Python package installation and SSL certificate verification errors on my local macOS environment, and structure the code to efficiently extract tree attributes (nodes, depth, features) from scikit-learn's underlying structures. I reviewed all generated code, tested it locally to ensure it ran successfully, and take full responsibility for the content of this repository.