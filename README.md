# Hill Climbing

Hill climbing is a mathematical optimization algorithm that belongs to the family of local search techniques. It is commonly used to iteratively improve a solution based on a cost or objective function. This package provides a simple implementation of the hill climbing algorithm and is useful for efficiently blending predictions from multiple machine learning models. The goal is to achieve an ensemble score that is better than that of any single model in the ensemble.

## How it works
Hill climbing starts with an initial solution, which is the predictions of one of the base models. It then iteratively explores neighboring solutions by adjusting the weights used to blend predictions from other models. If a new combination results in an improved value of the objective function, it becomes the current solution. This process repeats until no further improvement is possible, i.e. when a local optimum has been reached.

## Installation

```bash
pip install hill-climbing
```

## Example usage

```python
from hill_climbing import Climber
from sklearn.metrics import root_mean_squared_error


# Running hill climbing
climber = Climber(
    objective="minimize",
    eval_metric=root_mean_squared_error
)
climber.fit(X, y)

print(f"Best score: {climber.best_score}")
print(f"Best predictions: {climber.best_oof_preds}")

# Predicting on unseen data
test_preds = climber.predict(X_test)
```


## Example usage with cross-validation:
```python
from hill_climbing import ClimberCV
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import KFold


# Running hill climbing with CV
climber_cv = ClimberCV(
    objective="minimize",
    eval_metric=root_mean_squared_error,
    cv=KFold(n_splits=5, shuffle=True, random_state=42)
)
climber_cv.fit(X, y)

print(f"Best score: {climber_cv.best_score}")
print(f"Best predictions: {climber_cv.best_oof_preds}")

# Predicting on unseen data
test_preds = climber_cv.predict(X_test)
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `objective` | str | - | Either "maximize" or "minimize" the evaluation metric |
| `eval_metric` | callable | - | The evaluation metric function to optimize |
| `allow_negative_weights` | bool | False | Whether to allow negative weights. Note that allowing negative weights increases computation time, and in some cases may lead to overfitting |
| `precision` | float | 0.01 | Controls the step size when trying new weights. Lower values will lead to higher computation times  |
| `starting_model` | str | "best" | Starting model selection strategy ("best", "random", or one of the column names in `X`) |
| `score_decimal_places` | int | 3 | Number of decimal places for score display |
| `random_state` | int | 42 | Random seed for reproducibility |
| `verbose` | bool | True | Whether to output information during hill climbing |
| `n_jobs` | int | -1 | Number of parallel jobs (-1 means use all available cores) |
| `cv` | BaseCrossValidator | - | Cross-validation splitter from scikit-learn. This parameter is only available in `ClimberCV` |