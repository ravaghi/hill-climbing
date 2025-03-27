from sklearn.model_selection import BaseCrossValidator
from typing import Callable, Dict, List, Tuple, Union
from multiprocessing import Pool, cpu_count
from functools import partial
from abc import ABC
import pandas as pd
import numpy as np
import random
import time

from .utils import create_cli_bar_chart


class Climber(ABC):
    def __init__(
        self,
        objective: str,
        eval_metric: Callable,
        allow_negative_weights: bool = False,
        precision: float = 0.001,
        starting_model: str = "best",
        score_decimal_places: int = 3,
        random_state: int = 42,
        verbose: bool = True,
        n_jobs: int = -1,
    ):
        self.objective = objective
        self.eval_metric = eval_metric
        self.allow_negative_weights = allow_negative_weights
        self.precision = precision
        self.starting_model = starting_model
        self.verbose = verbose
        self.random_state = random_state
        self.n_jobs = cpu_count() if n_jobs == -1 else min(n_jobs, cpu_count())

        self._score_decimal_places = score_decimal_places
        self._weight_decimal_places = max(2, int(-np.log10(self.precision)))

        self.best_score = None
        self.best_oof_preds = None
        self.history = None

        self._validate_inputs()
        self._set_random_state()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._global_timer = time.time()

        X, y =self._validate_fit_inputs(X, y)
        
        weight_range = self._get_weight_range()
        model_scores = self._get_individual_model_scores(X, y)
        first_model = self._get_starting_model(model_scores)
        
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=self.objective == "maximize")
        sorted_models = [model for model, _ in sorted_models]
        X = X[sorted_models]
        
        self._print_header(model_scores)
        
        results = [first_model]
        coefs = [1]

        current_best_oof = X[first_model].values
        remaining_oof_preds = X.drop(first_model, axis=1)
        initial_score = self.eval_metric(y, current_best_oof)

        iteration = 0
        stop_climbing = False
        last_score = initial_score

        iteration_times = []

        self.history = pd.DataFrame([{
            "iteration": iteration,
            "model": first_model,
            "score": initial_score,
            "improvement": initial_score,
            "time": time.time() - self._global_timer
        }])

        if self.verbose:
            print(f"║{0:^11}║{first_model:^32}║{coefs[0]:^10.{self._weight_decimal_places}f}║{initial_score:^15.{self._score_decimal_places}f}║{'-':^15}║{'-':^10}║")

        while not stop_climbing and remaining_oof_preds.shape[1] > 0:
            start_time_iter = time.time()
            iteration += 1

            potential_best_score = self.eval_metric(y, current_best_oof) if self.objective == "maximize" else -self.eval_metric(y, current_best_oof)
            best_model, best_weight = None, None

            for model in remaining_oof_preds.columns:
                func_partial = partial(
                    self._compute_score,
                    current_preds=current_best_oof,
                    new_preds=remaining_oof_preds[model].values,
                    y_true=y
                )

                all_scores = self._parallelize_score_computation(func_partial, weight_range)
                for weight, score in all_scores:
                    if score > potential_best_score:
                        potential_best_score = score
                        best_model, best_weight = model, weight

            iter_time = time.time() - start_time_iter
            iteration_times.append(iter_time)

            if best_model is not None:
                results.append(best_model)
                coefs = [c * (1 - best_weight) for c in coefs] + [best_weight]
                current_best_oof = (1 - best_weight) * current_best_oof + best_weight * remaining_oof_preds[best_model].values
                remaining_oof_preds = remaining_oof_preds.drop(best_model, axis=1)

                current_score = self.eval_metric(y, current_best_oof)
                improvement = abs(current_score - last_score)
                improvement_str = f"{improvement:.{self._score_decimal_places}f}"
                if self.verbose:
                    print(f"║{iteration:^11}║{best_model:^32}║{best_weight:^10.{self._weight_decimal_places}f}║{current_score:^15.{self._score_decimal_places}f}║{improvement_str:^15}║{iter_time:^10.2f}║")

                last_score = current_score
                self.history = pd.concat([
                    self.history,
                    pd.DataFrame([{
                        "iteration": iteration,
                        "model": best_model,
                        "score": current_score,
                        "improvement": improvement,
                        "time": iter_time
                    }])
                ], ignore_index=True)
            else:
                stop_climbing = True
                
        self.history["coef"] = coefs

        self.best_score = last_score
        self.best_oof_preds = current_best_oof
        self._is_fitted = True

        self._print_final_results()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fit before making predictions")

        predictions = np.zeros(X.shape[0])
        for model, weight in zip(self.history["model"], self.history["coef"]):
            predictions += weight * X[model].values
            
        return predictions

    def _validate_inputs(self) -> None:
        if self.objective not in ["maximize", "minimize"]:
            raise ValueError("objective must be either 'maximize' or 'minimize'")
        
        if not callable(self.eval_metric):
            raise ValueError("eval_metric must be a callable function")
        
        if self.precision <= 0:
            raise ValueError("precision must be greater than 0")

    def _set_random_state(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)

    def _validate_fit_inputs(self, X: pd.DataFrame, y: Union[pd.Series, np.ndarray]) -> Tuple[pd.DataFrame, pd.Series]:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, (pd.Series, np.ndarray)):
            raise ValueError("y must be a pandas Series or numpy array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of rows")
        if X.shape[1] == 0:
            raise ValueError("X must have at least one column")
        
        if isinstance(y, np.ndarray):
            y = pd.Series(y)
            
        return X, y
        
    def _get_weight_range(self) -> np.ndarray:
        return np.arange(-0.5, 0.51, self.precision) if self.allow_negative_weights else np.arange(self.precision, 0.51, self.precision)

    def _get_individual_model_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        return {
            model: self.eval_metric(y, X[model].values)
            for model in X.columns
        }

    def _get_starting_model(self, model_scores: Dict[str, float]) -> str:
        if self.starting_model == "best":
            return max(model_scores.items(), key=lambda x: x[1] if self.objective == "maximize" else -x[1])[0]
        elif self.starting_model == "random":
            return random.choice(list(model_scores.keys()))
        return self.starting_model
        
    def _print_header(self, model_scores: Dict[str, float]) -> None:
        if not self.verbose:
            return

        table_width = 100
        print(f"╔{'═' * (table_width-2)}╗")
        print(f"║{'Configuration':^{table_width-2}}║")
        print(f"╚{'═' * (table_width-2)}╝")

        info = [
            ("Metric", self.eval_metric.__name__),
            ("Objective", self.objective),
            ("Precision", self.precision),
            ("Allow negative weights", self.allow_negative_weights),
            ("Starting model", self.starting_model),
            ("Number of parallel jobs", self.n_jobs),
            ("Number of models", len(model_scores))
        ]

        for label, value in info:
            label_text = f" {label}"
            value_text = str(value)
            available_width = table_width - 2 - len(label_text) - 1
            dots = "." * (available_width - len(value_text))
            print(f"{label_text} {dots} {value_text}")

        print(f"╔{'═' * 31}╦{'═' * 11}╦{'═' * 54}╗")
        print(f"║{'Model':^31}║{'Score':^11}║{'Performance Bar':^54}║")
        print(f"╠{'═' * 31}╬{'═' * 11}╬{'═' * 54}╣")
        for line in create_cli_bar_chart(model_scores, self.objective, self._score_decimal_places):
            parts = line.split(' | ')
            if len(parts) == 3:
                model, score, bar = parts
                bar_padded = bar + ' ' * (54 - len(bar))
                print(f"║{model:^31}║{score:^11}║{bar_padded}║")
        print(f"╚{'═' * 31}╩{'═' * 11}╩{'═' * 54}╝")

        print(f"\n\n╔{'═' * (table_width-2)}╗")
        print(f"║{'Running Hill Climbing':^{table_width-2}}║")
        print(f"╚{'═' * (table_width-2)}╝")

        print(f"╔{'═' * 11}╦{'═' * 32}╦{'═' * 10}╦{'═' * 15}╦{'═' * 15}╦{'═' * 10}╗")
        print(f"║{'Iteration':^11}║{'Model Added':^32}║{'Weight':^10}║{'Score':^15}║{'Improvement':^15}║{'Time (s)':^10}║")
        print(f"╠{'═' * 11}╬{'═' * 32}╬{'═' * 10}╬{'═' * 15}╬{'═' * 15}╬{'═' * 10}╣")

    def _compute_score(
        self,
        weight: float,
        current_preds: np.ndarray,
        new_preds: np.ndarray,
        y_true: np.ndarray
    ) -> Tuple[float, float]:
        ensemble_preds = (1 - weight) * current_preds + weight * new_preds
        score = self.eval_metric(y_true, ensemble_preds)
        return (weight, score) if self.objective == "maximize" else (weight, -score)

    def _parallelize_score_computation(
        self,
        func: Callable,
        weight_range: np.ndarray
    ) -> List[Tuple[float, float]]:
        num_cores = min(self.n_jobs, len(weight_range))
        with Pool(num_cores) as pool:
            return pool.map(func, weight_range)

    def _print_final_results(self) -> None:
        if not self.verbose:
            return

        print(f"╚{'═' * 11}╩{'═' * 32}╩{'═' * 10}╩{'═' * 15}╩{'═' * 15}╩{'═' * 10}╝")

        table_width = 100
        print(f"\n\n╔{'═' * (table_width-2)}╗")
        print(f"║{'Results':^{table_width-2}}║")
        print(f"╚{'═' * (table_width-2)}╝")

        summary_info = [
            ("Number of models in ensemble", f"{len(self.history)}")
        ]

        if len(self.history) > 1:
            improvement = abs(self.history["score"].iloc[-1] - self.history["score"].iloc[0])
            improvement_pct = improvement / abs(self.history["score"].iloc[0]) * 100 if self.history["score"].iloc[0] != 0 else 0
            improvement_sign = "+" if improvement > 0 else ""
            summary_info.append((
                "Overall improvement",
                f"{improvement_sign}{improvement:.{self._score_decimal_places}f} ({improvement_sign}{improvement_pct:.2f}%)"
            ))

        total_time = time.time() - self._global_timer
        iteration_times = self.history["time"].values
        summary_info.extend([
            ("Total time", f"{total_time:.2f} seconds"),
            ("Average iteration time", f"{sum(iteration_times) / len(iteration_times):.2f} seconds"),
            ("Final score", f"{self.best_score:.{self._score_decimal_places}f}")
        ])

        for label, value in summary_info:
            label_text = f" {label}"
            value_text = str(value)
            available_width = table_width - 2 - len(label_text) - 1
            dots = "." * (available_width - len(value_text))
            print(f"{label_text} {dots} {value_text}")


class ClimberCV(Climber):
    def __init__(self, cv: BaseCrossValidator, **kwargs):
        super().__init__(**kwargs)
        self.cv = cv
        self.fold_scores = []

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        self._global_timer = time.time()

        X, y = self._validate_fit_inputs(X, y)
        
        weight_range = self._get_weight_range()
        model_scores = self._get_individual_model_scores(X, y)
        
        self._print_header(model_scores)

        histories = []
        oof_preds = np.zeros(X.shape[0])
        for fold_idx, (train_index, val_index) in enumerate(self.cv.split(X, y)):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            val_scores = self._get_individual_model_scores(X_val, y_val)
            first_model = self._get_starting_model(val_scores)
            
            sorted_models = sorted(val_scores.items(), key=lambda x: x[1], reverse=self.objective == "maximize")
            sorted_models = [model for model, _ in sorted_models]
            X_train = X_train[sorted_models]
            X_val = X_val[sorted_models]
            
            results = [first_model]
            coefs = [1]

            best_train = X_train[first_model].values
            best_val = X_val[first_model].values
            remaining_train = X_train.drop(first_model, axis=1)
            remaining_val = X_val.drop(first_model, axis=1)
            
            initial_train_score = self.eval_metric(y_train, best_train)
            initial_val_score = self.eval_metric(y_val, best_val)

            iteration = 0
            stop_climbing = False

            iteration_times = []
            history = pd.DataFrame([{
                "iteration": iteration,
                "model": first_model,
                "train_score": initial_train_score,
                "val_score": initial_val_score,
                "time": time.time() - self._global_timer
            }])

            if self.verbose:
                print(f"║{fold_idx:^6}║{iteration:^7}║{first_model:^33}║{coefs[0]:^10.{self._weight_decimal_places}f}║{initial_train_score:^13.{self._score_decimal_places}f}║{initial_val_score:^13.{self._score_decimal_places}f}║{'-':^10}║")

            while not stop_climbing and remaining_train.shape[1] > 0:
                iteration += 1
                start_time_iter = time.time()
                
                potential_best_train_score = self.eval_metric(y_train, best_train) if self.objective == "maximize" else -self.eval_metric(y_train, best_train)
                potential_best_val_score = self.eval_metric(y_val, best_val) if self.objective == "maximize" else -self.eval_metric(y_val, best_val)
                best_model, best_weight = None, None
                for model in remaining_train.columns:
                    func_partial = partial(
                        self._compute_score,
                        current_preds=best_train,
                        new_preds=remaining_train[model].values,
                        y_true=y_train
                    )

                    all_scores = self._parallelize_score_computation(func_partial, weight_range)
                    for weight, score in all_scores:
                        if score > potential_best_train_score:
                            potential_best_train_score = score
                            potential_best_val_score = self.eval_metric(y_val, (1 - weight) * best_val + weight * remaining_val[model].values)
                            best_model, best_weight = model, weight

                iter_time = time.time() - start_time_iter
                iteration_times.append(iter_time)

                if best_model is not None:
                    results.append(best_model)
                    coefs = [c * (1 - best_weight) for c in coefs] + [best_weight]
                    best_train = (1 - best_weight) * best_train + best_weight * remaining_train[best_model].values
                    best_val = (1 - best_weight) * best_val + best_weight * remaining_val[best_model].values
                    remaining_train = remaining_train.drop(best_model, axis=1)
                    remaining_val = remaining_val.drop(best_model, axis=1)

                    if self.verbose:
                        print(f"║{fold_idx:^6}║{iteration:^7}║{best_model:^33}║{best_weight:^10.{self._weight_decimal_places}f}║{potential_best_train_score:^13.{self._score_decimal_places}f}║{potential_best_val_score:^13.{self._score_decimal_places}f}║{iter_time:^10.2f}║")

                    history = pd.concat([
                        history,
                        pd.DataFrame([{
                            "iteration": iteration,
                            "model": best_model,
                            "train_score": potential_best_train_score if self.objective == "maximize" else -potential_best_train_score,
                            "val_score": potential_best_val_score,
                            "time": iter_time
                        }])
                    ], ignore_index=True)
                    
                else:
                    stop_climbing = True
                    
            history["coef"] = coefs
            history["fold"] = fold_idx
            histories.append(history)

            oof_preds[val_index] = np.zeros(X_val.shape[0])
            for model, weight in zip(history["model"], history["coef"]):
                oof_preds[val_index] += weight * X_val[model].values

            self.fold_scores.append(self.eval_metric(y_val, oof_preds[val_index]))
            
            if self.verbose and fold_idx != self.cv.n_splits - 1:
                print(f"║{'-' * 6}║{'-' * 7}║{'-' * 33}║{'-' * 10}║{'-' * 13}║{'-' * 13}║{'-' * 10}║")
                
        self.history = pd.concat(histories)
        self.best_oof_preds = oof_preds
        self.best_score = self.eval_metric(y, oof_preds)
        self._is_fitted = True

        self._print_final_results()
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._is_fitted:
            raise ValueError("Model must be fit before making predictions")

        preds = np.zeros(X.shape[0])
        for i in range(self.cv.n_splits):
            fold_preds = np.zeros(X.shape[0])
            for model, weight in zip(self.history[self.history["fold"] == i]["model"], self.history[self.history["fold"] == i]["coef"]):
                fold_preds += weight * X[model].values
            preds += fold_preds
        return preds / self.cv.n_splits

    def _print_header(self, model_scores: Dict[str, float]) -> None:
        if not self.verbose:
            return

        table_width = 100
        print(f"╔{'═' * (table_width-2)}╗")
        print(f"║{'Configuration':^{table_width-2}}║")
        print(f"╚{'═' * (table_width-2)}╝")

        info = [
            ("Metric", self.eval_metric.__name__),
            ("Objective", self.objective),
            ("Precision", self.precision),
            ("Allow negative weights", self.allow_negative_weights),
            ("Starting model", self.starting_model),
            ("Number of parallel jobs", self.n_jobs),
            ("Number of models", len(model_scores)),
            ("Number of folds", self.cv.n_splits)
        ]

        for label, value in info:
            label_text = f" {label}"
            value_text = str(value)
            available_width = table_width - 2 - len(label_text) - 1
            dots = "." * (available_width - len(value_text))
            print(f"{label_text} {dots} {value_text}")

        print(f"╔{'═' * 32}╦{'═' * 10}╦{'═' * 54}╗")
        print(f"║{'Model':^32}║{'Score':^10}║{'Performance Bar':^54}║")
        print(f"╠{'═' * 32}╬{'═' * 10}╬{'═' * 54}╣")

        for line in create_cli_bar_chart(model_scores, self.objective, self._score_decimal_places):
            parts = line.split(' | ')
            if len(parts) == 3:
                model, score, bar = parts
                bar_padded = bar + ' ' * (54 - len(bar))
                print(f"║{model:^32}║{score:^10}║{bar_padded}║")

        print(f"╚{'═' * 32}╩{'═' * 10}╩{'═' * 54}╝")

        print(f"\n\n╔{'═' * (table_width-2)}╗")
        print(f"║{'Running Hill Climbing':^{table_width-2}}║")
        print(f"╚{'═' * (table_width-2)}╝")

        print(f"╔{'═' * 6}╦{'═' * 7}╦{'═' * 33}╦{'═' * 10}╦{'═' * 13}╦{'═' * 13}╦{'═' * 10}╗")
        print(f"║{'Fold':^6}║{'Itr':^7}║{'Model Added':^33}║{'Weight':^10}║{' Train score':^13}║{'Val score':^13}║{'Time (s)':^10}║")
        print(f"╠{'═' * 6}╬{'═' * 7}╬{'═' * 33}╬{'═' * 10}╬{'═' * 13}╬{'═' * 13}╬{'═' * 10}╣")

    def _print_final_results(self) -> None:
        if not self.verbose:
            return

        print(f"╚{'═' * 6}╩{'═' * 7}╩{'═' * 33}╩{'═' * 10}╩{'═' * 13}╩{'═' * 13}╩{'═' * 10}╝")

        table_width = 100
        print(f"\n\n╔{'═' * (table_width-2)}╗")
        print(f"║{'Results':^{table_width-2}}║")
        print(f"╚{'═' * (table_width-2)}╝")

        summary_info = []

        total_time = time.time() - self._global_timer
        iteration_times = self.history["time"].values
        summary_info.extend([
            ("Total time", f"{total_time:.2f} seconds"),
            ("Average iteration time", f"{sum(iteration_times) / len(iteration_times):.2f} seconds"),
            ("Final score", f"{self.best_score:.{self._score_decimal_places}f}")
        ])

        for label, value in summary_info:
            label_text = f" {label}"
            value_text = str(value)
            available_width = table_width - 2 - len(label_text) - 1
            dots = "." * (available_width - len(value_text))
            print(f"{label_text} {dots} {value_text}")
