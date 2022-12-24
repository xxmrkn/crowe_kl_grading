def prepare_train_df(train_file, fold, df): 

    #prepare dataframe for each fold
    with open(train_file[fold]) as f:
        line = f.read().splitlines()
    print(line)

    train_df = df[df['UID'].isin(line)]
    train_df = train_df.reset_index(drop=True)
    print('train_df',train_df)

    return train_df    

def prepare_valid_df(valid_file, fold, df): 

    #prepare dataframe for each fold
    with open(valid_file[fold]) as f:
        line2 = f.read().splitlines()
    print(line2)

    valid_df = df[df['UID'].isin(line2)]
    valid_df = valid_df.reset_index(drop=True)
    print('valid_df',valid_df)

    return valid_df    

def prepare_test_df(test_file, fold, df): 

    #prepare dataframe for each fold
    with open(test_file[fold]) as f:
        line3 = f.read().splitlines()
    print(line3)

    test_df = df[df['UID'].isin(line3)]
    test_df = test_df.reset_index(drop=True)
    #print('test_df',test_df)

    return test_df    