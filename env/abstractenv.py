class AbstractEnv:
    def __init__(self):
        pass
    def random_x0(self):
        raise NotImplementedError
    def step(self, x, u):
        raise NotImplementedError
    def get_trajectory(self, model, env, x0s, T):
        raise NotImplementedError
    



