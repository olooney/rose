# rose

Experiments on learning the `Petals Around the Rose` rule from examples of five dice.

The core implementation lives in `src/rose.py`. It generates synthetic training data, trains several approaches, and evaluates how quickly each model learns the scoring rule. The approaches in this repo include linear baselines, a bincount feature model, a GAM, a gradient-boosted tree, a fully connected neural network, and a DeepSet model.

Most of the exploration happens in the notebooks under `notebooks/`. These cover concept learning, exact methods, linear feature ideas, tree and neural network experiments, hyperparameter tuning, and a DeepSet animation. Generated comparison tables and plots are saved under `docs/approaches/`, with the HTML summary template in `templates/summary.html`.

## Structure

- `src/rose.py`: data generation, models, evaluation, and plotting helpers
- `notebooks/`: exploratory analysis and experiment notebooks
- `docs/approaches/`: saved results, plots, and summary outputs
- `templates/summary.html`: HTML template for the report summary