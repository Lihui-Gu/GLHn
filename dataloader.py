import torch
import numpy as np
class DataLoader:
    def __init__(self, df_feature, df_label, df_stock_index, device = None):
        assert len(df_feature) == len(df_label)
        self.df_feature = df_feature.values
        self.df_label = df_label.values
        self.df_stock_index = df_stock_index
        self.index = df_label.index
        self.daily_count = df_label.groupby(level=0).size().values
        self.daily_index = np.roll(np.cumsum(self.daily_count), 1)
        self.daily_index[0] = 0

    @property
    def batch_length(self):
      return self.daily_length

    @property
    def daily_length(self):
        return len(self.daily_count)

    def iter_batch(self):
      yield from self.iter_daily_shuffle()
      return

    def iter_daily_shuffle(self):
        indices = np.arange(len(self.daily_count))
        np.random.shuffle(indices)
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])

    def iter_daily(self):
        indices = np.arange(len(self.daily_count))
        for i in indices:
            yield i, slice(self.daily_index[i], self.daily_index[i] + self.daily_count[i])
    def get(self, slc):
        outs = self.df_feature[slc], self.df_label[slc][:,0], self.df_stock_index[slc]
        return outs + (self.index[slc],)