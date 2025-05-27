import torch.nn as nn
import contextlib
#import contextlib

class AbstractModule(nn.Module):
    """
    A base class for PyTorch modules that provides an extendable context
    management interface.

    This abstract module behaves like a standard `nn.Module`, but includes
    a placeholder context manager `_enter_variable_scope` which can be
    overridden in subclasses to define custom behavior (e.g., scoping,
    logging, temporary parameter changes) within a controlled context.

    Attributes
    ----------
    Inherits all attributes from `torch.nn.Module`.

    Methods
    -------
    _enter_variable_scope(*args, **kwargs)
        A context manager that yields `None`. Designed to be overridden
        in derived classes to implement custom scoping or setup logic.
    """
    def __init__(self, *args, **kwargs):
        """
        Initializes the abstract module.

        Parameters
        ----------
        *args : tuple
            Positional arguments passed to `nn.Module`.
        **kwargs : dict
            Keyword arguments passed to `nn.Module`.
        """
        super(AbstractModule, self).__init__()

    @contextlib.contextmanager
    def _enter_variable_scope(self, *args, **kwargs):
        """
        A placeholder context manager to be optionally overridden by subclasses.

        Can be used to manage a variable scope or inject behavior that should
        happen before and after a block of code (e.g., parameter sharing,
        logging, or instrumentation).

        Yields
        ------
        None
        """
        yield None