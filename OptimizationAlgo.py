import pandas as pd
import numpy as np
import yfinance as yf
class OptimizationAlgorithm():
    """
    
    Attributes
    ----------
    symb : str
    slowmvAvg : int
    fastmvAvg : int
    start : str
    end : str
    data : pandas.DataFrame
    results : pandas.DataFrame
    
    Methods
    -------
    get_data():
    prepare_data():
    set_parameters(slowmvAvg=None, fastmvAvg=None):
    test_strategy():
    plot_results():

    optimize_parameters(slowmvAvg_range, fastmvAvg_range):

    
    """
    
    def __init__(self, symb, slowmvAvg, fastmvAvg, start, end):
        self.symb = symb
        self.slowmvAvg = slowmvAvg
        self.fastmvAvg = fastmvAvg
        self.start = start
        self.end = end
        self.data = None
        self.results = None
        self.get_data()
        self.prepare_data()
        
    
    def get_data(self):

        raw = yf.download(self.symb, start=self.start, end=self.end)
        raw['returns'] = np.log(raw['Close'] / raw['Close'].shift(1))
        self.data = raw[['Close', 'returns']].rename(columns={'Close': 'close'})

    def prepare_data(self):

        data = self.data.copy()
        data['slowmvAvg'] = data['close'].rolling(self.slowmvAvg).mean()
        data['fastmvAvg'] = data['close'].rolling(self.fastmvAvg).mean()
        self.data = data
    
    def set_parameters(self, slowmvAvg=None, fastmvAvg=None):

        data = self.data.copy()
        self.slowmvAvg = slowmvAvg
        self.fastmvAvg = fastmvAvg
        data = data.drop(['slowmvAvg', 'fastmvAvg'], axis=1)
        data['slowmvAvg'] = data['close'].rolling(self.slowmvAvg).mean()
        data['fastmvAvg'] = data['close'].rolling(self.fastmvAvg).mean()
        self.data = data
            
    def test_strategy(self):

        data = self.data.copy().dropna()
        data['position'] = np.where(data['slowmvAvg'] > data['fastmvAvg'], 1, -1) 
        data['strategy'] = data['position'].shift(1) * data['returns']
        data.dropna(inplace=True)
        data['creturns'] = data['returns'].cumsum().apply(np.exp)
        data['cstrategy'] = data['strategy'].cumsum().apply(np.exp)
        self.results = data
        
        #print(data)
        #print(data['cstrategy'])
        perf = data['cstrategy'].iloc[-1]
        outperf = perf - data['creturns'].iloc[-1]
        return round(perf, 6), round(outperf, 6)
    
    def plot_results(self):

        if self.results is None:
            print('Run test_strategy() first.')
        else:
            title = '{} | slowmvAvg = {} | fastmvAvg= {}'.format(self.symb, self.slowmvAvg, self.fastmvAvg)

            self.results[['creturns','cstrategy']].plot(title=title, figsize=(12,8))

    def optimize_parameters(self, slowmvAvg_range, fastmvAvg_range):
        combos = []
        for i in range(slowmvAvg_range[0],slowmvAvg_range[1]+slowmvAvg_range[2],slowmvAvg_range[2]):
            for j in range(fastmvAvg_range[0],fastmvAvg_range[1]+fastmvAvg_range[2],fastmvAvg_range[2]):
                combos.append((i,j))
        
        results = []

        for comb in combos:
            self.set_parameters(comb[0], comb[1])
            results.append(self.test_strategy()[0])


        most_optimal_results = np.max(results)
        opt = combos[np.argmax(results)]
        self.set_parameters(opt[0], opt[1])
        
        self.test_strategy()

        many_results = pd.DataFrame(data=combos, columns=['slowmvAvg', 'fastmvAvg'])
        many_results['performance'] = results
        self.results_overview = many_results

        return opt, most_optimal_results
