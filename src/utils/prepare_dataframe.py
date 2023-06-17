from dataclasses import dataclass

@dataclass
class PrepareDataframe(object):

    fold: int
    df: object
    
    def prepare_train_df(self, train_file): 

        # Prepare dataframe for each fold
        with open(train_file[self.fold]) as f:
            line = f.read().splitlines()

        train_df = self.df[self.df['UID'].isin(line)]
        train_df = train_df.reset_index(drop=True)

        return train_df  

    def prepare_valid_df(self, valid_file): 

        # Prepare dataframe for each fold
        with open(valid_file[self.fold]) as f:
            line = f.read().splitlines()

        valid_df = self.df[self.df['UID'].isin(line)]
        valid_df = valid_df.reset_index(drop=True)

        return valid_df    

    def prepare_test_df(self, test_file): 

        # Prepare dataframe for each fold
        with open(test_file[self.fold]) as f:
            line = f.read().splitlines()

        test_df = self.df[self.df['UID'].isin(line)]
        test_df = test_df.reset_index(drop=True)

        return test_df    