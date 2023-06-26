import pandas as pd
from dataclasses import dataclass

@dataclass
class DataframeHelper(object):

    df: pd.DataFrame
    fold: int = 4
    
    def prepare_df(self, file): 

        # Prepare dataframe for each fold
        with open(file[self.fold]) as f:
            line = f.read().splitlines()

        df = self.df[self.df['UID'].isin(line)]
        df = df.reset_index(drop=True)

        return df  