import numpy

from .kraken import Kraken



class Bootstrap:
    def __init__(self, df, method):
        self.df = df
        self.method = method

    def _iteration(self, iteration, n):
        idx = numpy.random.randint(0,self.df.shape[0],self.df.shape[0])
        df = self.df.iloc[idx]
        return self.method(df)

    def run(self, N=100):
        configs = [dict(n=i) for i in range(N)]
        return Kraken.release(self._iteration, configs)

