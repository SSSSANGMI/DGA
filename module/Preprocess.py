import pandas as pd
from publicsuffixlist import PublicSuffixList

class Preprocess():

    def __init__(self):
        self.result = 0

    
    @staticmethod
    def preprocessAtOnce(preprocessList, normal_df, dga_df):

        if 'defineColumns' in preprocessList:
            normal_df = Preprocess.basicColumns(normal_df)
            dga_df = Preprocess.basicColumns(dga_df)

        if 'removeNull' in preprocessList:    
            normal_df = Preprocess.removeNull(normal_df)
            dga_df = Preprocess.removeNull(dga_df)

            normal_df = normal_df.reset_index(drop=True)
            dga_df = dga_df.reset_index(drop=True)

        if 'removeNotDomain' in preprocessList:
            normal_df = Preprocess.removeNotDomain(normal_df)
            dga_df = Preprocess.removeNotDomain(dga_df)

            normal_df = normal_df.reset_index(drop=True)
            dga_df = dga_df.reset_index(drop=True)

        if 'removeSuffix' in preprocessList:
            psl = PublicSuffixList()
            normal_df['suffix'] = normal_df['domain'].apply(lambda x: Preprocess.ignoreVPS(psl, x))
            dga_df['suffix'] = dga_df['domain'].apply(lambda x: Preprocess.ignoreVPS(psl, x))


        return normal_df, dga_df
    
    
    @staticmethod
    def basicColumns(df) -> pd.DataFrame():
        df.columns = ['domain', 'label']

        return df

    @staticmethod
    def removeNull(df) -> pd.DataFrame():
        df = df.dropna(axis=0)
        df = df.reset_index(drop=True)

        return df

    ''' 
    Defination of Not Domain
    ['-', '--']
    '''
    @staticmethod
    def removeNotDomain(df) -> pd.DataFrame:
        for idx in range(len(df)):
            if df.loc[idx, 'domain']  == '--' or  df.loc[idx, 'domain'] == '-':
                df = df.drop(idx)
        return df


    @staticmethod
    def ignoreVPS(psl, domain) -> str:

        # Return the rest of domain after ignoring the Valid Public Suffixes:
        validPublicSuffix = '.' + psl.publicsuffix(domain)

        if len(validPublicSuffix) < len(domain):
            # If it has VPS
            subString = domain[0: domain.index(validPublicSuffix)]
        elif len(validPublicSuffix) == len(domain):
            return 0
        else:
            # If not
            subString = domain

        return subString

