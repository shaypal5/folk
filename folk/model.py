"""Parameterized model abstraction."""

# from sklearn.model_selection import P


class ParameterizedModel(object):
    """A parameterized model.

    Parameters
    ----------
    model_getter : function
        A function that returns an estimator object implementing ‘fit’ when
        supplied with values for its required keyword arguments, ignoring any
        unkown keyword arguments.
    param_grid : iterable over dict of string to any
        An iterable over parameter realizations. The ParameterGrid sklearn
        class is a great example.
    """

    def __init__(self, model_getter, param_grid):
        self.model_getter = model_getter
        self.param_grid = param_grid

    def __iter__(self):
        """Iterate on all realized models induced by this parameterized model.

        Returns
        -------
        models : iterator over sklearn.BaseEstimator objects
            Yields scikit-learn estimator objects.
        """
        for params in self.param_grid:
            yield self.model_getter(**params)

    def model_by_params(self, params):
        """Returns a realized model by the given params.

        Parameters
        ----------
        params : dict of string to any
            Parameter assignments by which to produce a model.

        Returns
        -------
        sklearn.BaseEstimator
            An estimator.
        """
        return self.model_getter(**params)
