import time

import pandas as pd

import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import confusion_matrix, plot_confusion_matrix

class RF_Model():
        
    def __init__(self):
        self.result = 0

    @staticmethod
    def divideTrainTest(df, test_ratio, RANDOM_SEED):
        attributes = df.drop('label', axis=1)
        observed = df['label']

        x_train, x_test, y_train, y_test = train_test_split(attributes, observed, test_size = test_ratio, random_state = RANDOM_SEED)
        return x_train, x_test, y_train, y_test


    # Visualization Performance of tuning hyperparameter
    @staticmethod
    def optimi_visualization(algorithm_name, x_values, train_score, test_score, xlabel, filename):
        data_path = '../data'

        plt.plot(x_values, train_score, linestyle = '-', label = 'train score')
        plt.plot(x_values, test_score, linestyle = '--', label = 'test score')

        plt.ylabel('Accuracy(%)')
        plt.xlabel(xlabel)
        plt.legend()
        plt.savefig(data_path + '/' + algorithm_name + '_' + filename + '.png')


    # TreeNum optimization
    @staticmethod
    def optimi_estimator(algorithm, x_train, y_train, x_test, y_test, n_estimator_min, n_estimator_max):
        train_score = [];
        test_time = [];
        test_pertime = [];
        test_precision = [];
        test_recall = [];
        test_cf = [];
        test_auc = []

        para_n_tree = [n_tree*5 for n_tree in range(n_estimator_min, n_estimator_max)]

        for v_n_estimators in para_n_tree:
            model = algorithm(n_estimators = v_n_estimators, random_state=1234)
            model.fit(x_train, y_train)

            start = time.time()
            test_pred = model.predict(x_train)
            end = time.time()

            train_score.append(model.score(x_train, y_train))
            test_time.append(end - start)
            test_pertime.append(len(x_test)/(end-start))
            test_precision.append(precision_score(y_test, test_pred))
            test_recall.append(recall_score(y_test, test_pred))
            test_cf.append(confusion_matrix(y_test, test_pred, labels=[1,0]))
            test_auc.append(format(roc_auc_score(y_test,test_pred)))

        df_score_n = pd.DataFrame({'n_tree': para_n_tree, 'TrainScore': train_score,
                                   'test_time': test_time, 'test_pertime': test_pertime, 'test_precision': test_precision, 'test_recall': test_recall,
                                   'test_cf': test_cf, 'test_auc':test_auc})

        return df_score_n


    # depth estimation
    @staticmethod
    def optimi_maxdepth (algorithm, x_train, y_train, x_test, y_test, depth_min, depth_max, n_estimator):
        train_score = [];
        test_time = [];
        test_pertime = [];
        test_precision = [];
        test_recall = [];
        test_cf = [];
        test_auc = []

        para_depth = [depth for depth in range(depth_min, depth_max)]

        for v_max_depth in para_depth:

            model = algorithm(max_depth = v_max_depth,
                                n_estimators = n_estimator,
                                random_state=1234)

            model.fit(x_train, y_train)

            start = time.time()
            test_pred = model.predict(x_train)
            end = time.time()

            train_score.append(model.score(x_train, y_train))
            test_time.append(end - start)
            test_pertime.append(len(x_test)/(end-start))
            test_precision.append(precision_score(y_test, test_pred))
            test_recall.append(recall_score(y_test, test_pred))
            test_cf.append(confusion_matrix(y_test, test_pred, labels=[1,0]))
            test_auc.append(format(roc_auc_score(y_test,test_pred)))

        df_score_n = pd.DataFrame({'depth': para_depth, 'TrainScore': train_score,
                                   'test_time': test_time, 'test_pertime': test_pertime, 'test_precision': test_precision, 'test_recall': test_recall,
                                   'test_cf': test_cf, 'test_auc':test_auc})

        return df_score_n




    #Optimization row split for node
    @staticmethod
    def optimi_minsplit (algorithm, algorithm_name, x_train, y_train, x_test, y_test, n_split_min, n_split_max, n_estimator, n_depth):
        train_score = [];
        test_time = [];
        test_pertime = [];
        test_precision = [];
        test_recall = [];
        test_cf = [];
        test_auc = []

        para_split = [n_split*2 for n_split in range(n_split_min, n_split_max)]
        for v_min_samples_split in para_split:


            model = algorithm(min_samples_split = v_min_samples_split,
                              n_estimators = n_estimator,
                              max_depth = n_depth,
                              random_state = 1234)

            model.fit(x_train, y_train)

            start = time.time()
            test_pred = model.predict(x_train)
            end = time.time()

            train_score.append(model.score(x_train, y_train))
            test_time.append(end - start)
            test_pertime.append(len(x_test)/(end-start))
            test_precision.append(precision_score(y_test, test_pred))
            test_recall.append(recall_score(y_test, test_pred))
            test_cf.append(confusion_matrix(y_test, test_pred, labels=[1,0]))
            test_auc.append(format(roc_auc_score(y_test,test_pred)))

        df_score_n = pd.DataFrame({'min_samples_split': para_split, 'TrainScore': train_score,
                                   'test_time': test_time, 'test_pertime': test_pertime, 'test_precision': test_precision, 'test_recall': test_recall,
                                   'test_cf': test_cf, 'test_auc':test_auc})

        return df_score_n



    #min-leaf optimization
    @staticmethod
    def optimi_minleaf(algorithm, x_train, y_train, x_test, y_test, n_leaf_min, n_leaf_max, n_estimator, n_depth, n_split):
        train_score = [];
        test_time = [];
        test_pertime = [];
        test_precision = [];
        test_recall = [];
        test_cf = [];
        test_auc = []

        para_leaf = [n_leaf*2 for n_leaf in range(n_leaf_min, n_leaf_max)]

        for v_min_samples_leaf in para_leaf:

            model = algorithm(min_samples_leaf = v_min_samples_leaf,
                              n_estimators = n_estimator,
                              max_depth = n_depth,
                              min_samples_split = n_split,
                              random_state=1234)
            model.fit(x_train, y_train)

            start = time.time()
            test_pred = model.predict(x_train)
            end = time.time()

            train_score.append(model.score(x_train, y_train))
            test_time.append(end - start)
            test_pertime.append(len(x_test)/(end-start))
            test_precision.append(precision_score(y_test, test_pred))
            test_recall.append(recall_score(y_test, test_pred))
            test_cf.append(confusion_matrix(y_test, test_pred, labels=[1,0]))
            test_auc.append(format(roc_auc_score(y_test,test_pred)))

        df_score_n = pd.DataFrame({'min_samples_leaf': para_leaf, 'TrainScore': train_score,
                                   'test_time': test_time, 'test_pertime': test_pertime, 'test_precision': test_precision, 'test_recall': test_recall,
                                   'test_cf': test_cf, 'test_auc':test_auc})

        return df_score_n

    # final model
    @staticmethod
    def model_final(algorithm, x_train, y_train, x_test, y_test, n_estimator, n_depth, n_split, n_leaf):

        
        model = algorithm(random_state = 1234,
                          n_estimators = n_estimator,
                          #                           min_samples_leaf = n_leaf,
                          #                           min_samples_split = n_split,
                          max_depth = n_depth)


        model.fit(x_train, y_train)

        start = time.time()
        test_pred = model.predict(x_test)
        end = time.time()

        model_path = './model/'
        model_filename = 'dga_rf' + '.pkl'
        with open(model_path + model_filename, 'wb') as f:
            pickle.dump(model, f)
        print(f"최종 모델 저장 완료! 파일 경로: {model_path + model_filename}\n")

        # Estimation
        print(f"train-Accuracy: {model.score(x_train, y_train):.4f}")
        print(f"running time: {end - start}")
        print(f"classification per second:, {len(x_test)/(end-start):.4f}")
        print(f"precision:, {precision_score(y_test, test_pred):.4f}")
        print(f"recall, {recall_score(y_test, test_pred):.4f}")
        print(f"confusion matrix\n,{ confusion_matrix(y_test, test_pred, labels=[1,0])}")
        print(f"F1-score: {f1_score(y_test, test_pred):.4f}")
        print(f"AUC, {format(roc_auc_score(y_test,test_pred))}")

        
        plt.figure(figsize =(30, 30))
        plot_confusion_matrix(model,
                              x_test, y_test,
                              include_values = True,
                              display_labels = ['dga', 'normal'], # 목표변수 이름
                              cmap = 'Pastel1') # 컬러맵
        plt.savefig('./result' + '/rf_confusion_matrix.png') # 혼동행렬 자료 저장
        plt.show()


        dt_importance = pd.DataFrame()
        feature_name = x_train.columns
        dt_importance['Feature'] = feature_name # 설명변수 이름
        dt_importance['Importance'] = model.feature_importances_ # 설명변수 중요도 산출


        dt_importance.sort_values("Importance", ascending = False, inplace = True)
        print(dt_importance.round(3))

        dt_importance.sort_values("Importance", ascending = True, inplace = True)

        coordinates = range(len(dt_importance)) 
        plt.barh(y = coordinates, width = dt_importance["Importance"])
        plt.yticks(coordinates, dt_importance["Feature"])
        plt.xlabel("Feature Importance") 
        plt.ylabel("Features")
        plt.savefig('./result' + '/Ramdomforest_feature_importance.png') # 변수 중요도 그래프 저장


