import argparse
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from Module.Preprocess import *
from Module.FeatureExtraction import *
from Module.RF_Model import *
from Module.XGB_Model import *

RANDOM_SEED = 1

def main():
    ''' PARAMETER
    '''
    data_path = './data'
    result_path = './result'
    normal_file = '/normal.csv'
    dga_file = '/dga.csv'


    ''' LOAD DATA
    INPUT: normal_file, dga_file
        method1. prompt with parameter
        method2. IDE with filepath code
    '''
    parser = argparse.ArgumentParser(description="DGA CLASSIFICATION BY Random Forest")
    parser.add_argument("-n", "--normal", help="NORMAL file")
    parser.add_argument("-d", "--dga", help="DGA file")
    args = parser.parse_args()


    if (args.normal == None) or (args.dga == None):
        normal_df = pd.read_csv(data_path + normal_file)
        dga_df = pd.read_csv(data_path + dga_file)
    else:
        normal_df = pd.read_csv(args.normal)
        dga_df = pd.read_csv(args.dga)

    ''' PREPROCESS
    1. define columns
    2. remove Null
    3. remove not-domain ('-', '--')
    4. remove suffix
    '''

    preprocess = Preprocess()
    preprocessList = ['defineColumns', 'removeNull', 'removeNotDomain', 'removeSuffix']
    normal_df, dga_df = preprocess.preprocessAtOnce(preprocessList, normal_df, dga_df)

    domain_df = pd.concat([normal_df, dga_df])
    domain_df.columns = ['domain', 'label', 'suffixed']
    domain_df = domain_df.reset_index(drop=True)
    print(domain_df)


    domain_shuffle_df = domain_df.sample(frac = 1, random_state=RANDOM_SEED)
    domain_shuffle_df.to_csv(result_path + '/domain_preprocess.csv', index = False)

    '''FEATURE EXTRACTION
    features: ['DNL', 'NoS','SLM','HwP','HVTLD','CSCS','CTS','UR','CIPA','contains_digit',
                   'vowel_ratio','digit_ratio','RRC','RCC','RCD','Entropy']
    '''
    # domain_df = domain_shuffle_df.copy()
    domain_df = pd.read_csv(result_path + '/domain_preprocess.csv')

    featureList = ['DNL', 'NoS','SLM','HwP','HVTLD','CSCS','CTS','UR','CIPA','contains_digit',
                   'vowel_ratio','digit_ratio','RRC','RCC','RCD','Entropy']
    suffixed = True
    domain_withFeatures = FeatureExtraction.featureExtractionAtOnce(featureList, domain_df, suffixed)
    FeatureExtraction.corr_features(domain_withFeatures)

    domain_withFeatures.to_csv(result_path +'/domain_withFeatures.csv', index=False)
    domain_withFeatures = pd.read_csv(result_path + '/domain_withFeatures.csv')

    drop_columns = {'UR', 'CIPA'}
    domain_withFeatures_filtered = FeatureExtraction.drop_features(domain_withFeatures, drop_columns)

    domain_withFeatures_filtered.to_csv(result_path + '/domain_withFeatures_filtered.csv', index=False)
    pd.set_option('display.max_columns', None) ## 모든 열을 출력한다
    feature = pd.read_csv(result_path + '/domain_withFeatures_filtered.csv')
    print(feature)


    '''Modeling and Optimization
    1. Random Forset
    2, (alpha) XGBoost
    '''
    algorithm = XGBClassifier # RandomForestClassifier / XGBClassifier
    # domain_df = domain_withFeatures_filtered.copy()
    domain_df = pd.read_csv(result_path + '/domain_withFeatures_filtered.csv')
    drop_columns = {'domain', 'suffixed'}
    domain_df = FeatureExtraction.drop_features(domain_df, drop_columns)

    test_ratio = 0.2

    x_train, x_test, y_train, y_test = RF_Model.divideTrainTest(domain_df, test_ratio, RANDOM_SEED)

    if algorithm == RandomForestClassifier:
        n_estimator = 20
        n_depth = 30

        RF_Model.model_final(algorithm, x_train, y_train, x_test, y_test, n_estimator, n_depth, 0, 0)

    elif algorithm == XGBClassifier:
        n_estimator = 1000
        max_depth = 6
        learning_rate = 0.5

        algorithm = XGBClassifier

        XGB_Model.model_final(algorithm, x_train, y_train, x_test, y_test, n_estimator, max_depth, learning_rate)


if __name__ == "__main__":
    main()
