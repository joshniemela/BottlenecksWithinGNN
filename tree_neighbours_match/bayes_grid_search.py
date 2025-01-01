from functools import partial
from itertools import product
from bayes_opt import BayesianOptimization


# Grid params is a dict of sets of values, we want to return a list of all possible combinations
def unwrap_grid_params(grid_params):
    expanded_params = {}
    for key, value in grid_params.items():
        if isinstance(value, tuple) and len(value) == 2:
            # This allows us to define a range, eg (2, 5) will be expanded to [2, 3, 4, 5]
            expanded_params[key] = range(value[0], value[1] + 1)
        else:
            expanded_params[key] = value

    keys = list(expanded_params.keys())
    values = [list(v) for v in expanded_params.values()]
    combinations = [dict(zip(keys, combination)) for combination in product(*values)]
    return combinations


def run_bayes_search(
    objective,
    partial_params: dict,
    p_bounds,
    n_random_iters=10,
    n_bayes_iters=20,
    random_state=42,
):
    bo = BayesianOptimization(
        f=partial(objective, **partial_params),
        pbounds=p_bounds,
        random_state=random_state,
        verbose=2,
    )
    bo.maximize(init_points=n_random_iters, n_iter=n_bayes_iters)
    max_bayes = bo.max
    assert max_bayes is not None
    max_bayes["params"].update(partial_params)

    return max_bayes


class BayesGridSearch:
    def __init__(self, grid_params, p_bounds, n_random_iters=10, n_bayes_iters=20):
        # These are the grid params we want to search over
        self.grid_combinations = unwrap_grid_params(grid_params)
        self.p_bounds = p_bounds
        self.n_random_iters = n_random_iters
        self.n_bayes_iters = n_bayes_iters

    def run(self, objective):
        for partial_params in self.grid_combinations:
            yield run_bayes_search(
                objective=objective,
                partial_params=partial_params,
                p_bounds=self.p_bounds,
                n_random_iters=self.n_random_iters,
                n_bayes_iters=self.n_bayes_iters,
            )
