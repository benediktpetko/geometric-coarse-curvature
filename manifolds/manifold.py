class RiemannianManifold:
    def __init__(self, dim: int):
        self.dim = dim

    def sample(self):
        pass


class EmbeddedManifold(RiemannianManifold):
    def __init__(self, dim: int, ambient_dim: int):
        super().__init__(dim)
        self.ambient_dim = ambient_dim
