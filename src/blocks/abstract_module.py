

class AbstractModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super(AbstractModule, self).__init__()

    @contextlib.contextmanager
    def _enter_variable_scope(self, *args, **kwargs):
        yield None