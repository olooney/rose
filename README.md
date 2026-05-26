# Machine Learning Approaches to Petals Around the Rose

This is a collection of experiments for automated algorithms that can solve the `Petals Around the Rose` game and correctly deduce
the "rule" used to count the number of "petals" on a roll of five dice. These are didactic, intended to showcase weaker and stronger
methods, so any particular method might be intentionally sub-optimal.

The core implementation lives in `src/rose/`. It generates synthetic training data, trains several approaches, and evaluates how quickly each model learns the scoring rule. The approaches in this repo include linear baselines, a bincount feature model, a GAM, a gradient-boosted tree, a fully connected neural network, and a DeepSet model.

Most of the exploration happens in the notebooks under `notebooks/`. These cover concept learning, exact methods, linear feature ideas, tree and neural network experiments, hyperparameter tuning, and a DeepSet animation. Generated comparison tables and plots are saved under `docs/approaches/`, with the HTML summary template in `templates/summary.html`.

## Setup

Install the project in editable mode with development dependencies:

```bash
pip install -e .[dev]
```

Then install the git hooks:

```bash
pre-commit install
```

## Tests

Run the test suite with:

```bash
pytest
```

## Structure

```text
rose/
|-- docs/
|-- images/
|-- notebooks/
|-- src/
|   `-- rose/
|-- templates/
`-- tests/
```

- `src/rose/`: data generation, approaches, evaluation, and plotting helpers
- `notebooks/`: exploratory analysis and experiment notebooks
- `docs/approaches/`: saved results, plots, and summary outputs
- `templates/summary.html`: HTML template for the report summary
- `tests/`: smoke and unit tests
