import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC,LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.tree  import DecisionTreeClassifier
from sklearn.neural_network import  MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import time


def timeCost(func):
    def _deco(self):
        start = time.time()
        print('当前模型: ',func.__name__)
        func(self)
        dur = time.time() - start

        if dur < 60:
            print('耗时: %.4fs' % dur)
        else:
            mins = int(dur // 60)
            sec = dur % 60
            print('耗时: {}分{:.3f}秒'.format(mins, sec))

        print('-------------------------------->')
    return _deco


class EDA(object):
    '''
        init同时进行数据的划分
    '''
    def __init__(self,_origin,_test,y):
        self.origin_data = pd.read_table(_origin, header=None, sep=',').values
        self.X_test = pd.read_table(_test, header=None, sep=',').values
        self.y_test = pd.read_table(y,header=None).values

        self.X_origin_data, self.y_origin_data = self.origin_data[:, :-1], self.origin_data[:, -1]

        self.X_train, self.X_valid, self.y_train, self.y_valid = \
            train_test_split(self.X_origin_data, self.y_origin_data, test_size=0.25,
                                                              random_state=0)


    @timeCost
    def LR(self):
        clf = LogisticRegression(max_iter=200).fit(self.X_train,self.y_train)
        # clf.predict(self.X_valid)

        print('测试集: ',clf.score(self.X_valid,self.y_valid))
        print('训练集: ',clf.score(self.X_test,self.y_test))


    @timeCost
    def LinearSVM(self):
        clf = LinearSVC(max_iter=1000).fit(self.X_train, self.y_train)

        print('测试集: ', clf.score(self.X_valid, self.y_valid))
        print('训练集: ', clf.score(self.X_test, self.y_test))


    @timeCost
    def KNN(self):
        clf = KNeighborsClassifier(n_neighbors=5).fit(self.X_train, self.y_train)

        print('测试集: ', clf.score(self.X_valid, self.y_valid))
        print('训练集: ', clf.score(self.X_test, self.y_test))


    @timeCost
    def gaussian(self):
        clf = GaussianNB().fit(self.X_train, self.y_train)

        print('测试集: ', clf.score(self.X_valid, self.y_valid))
        print('训练集: ', clf.score(self.X_test, self.y_test))

    # @timeCost
    # def multinomia(self):
    #     clf = MultinomialNB().fit(self.X_train, self.y_train)
    #
    #     print('测试集: ', clf.score(self.X_valid, self.y_valid))
    #     print('训练集: ', clf.score(self.X_test, self.y_test))

    @timeCost
    def bernouli(self):
        clf = BernoulliNB().fit(self.X_train, self.y_train)

        print('测试集: ', clf.score(self.X_valid, self.y_valid))
        print('训练集: ', clf.score(self.X_test, self.y_test))


    @timeCost
    def DT(self):
        clf = DecisionTreeClassifier().fit(self.X_train, self.y_train)

        print('测试集: ', clf.score(self.X_valid, self.y_valid))
        print('训练集: ', clf.score(self.X_test, self.y_test))

    @timeCost
    def Neural(self):
        clf = MLPClassifier().fit(self.X_train, self.y_train)

        print('测试集: ', clf.score(self.X_valid, self.y_valid))
        print('训练集: ', clf.score(self.X_test, self.y_test))

    @timeCost
    def RF(self):
        clf = RandomForestClassifier().fit(self.X_train, self.y_train)

        print('测试集: ', clf.score(self.X_valid, self.y_valid))
        print('训练集: ', clf.score(self.X_test, self.y_test))

    @timeCost
    def GBDT(self):
        clf = GradientBoostingClassifier().fit(self.X_train, self.y_train)

        print('测试集: ', clf.score(self.X_valid, self.y_valid))
        print('训练集: ', clf.score(self.X_test, self.y_test))


if __name__ == "__main__":
    origin_data = 'data/train_data.txt'
    test_data = 'data/test_data.txt'
    test_label = 'projects/student/answer.txt'

    eda = EDA(origin_data,test_data,test_label)
    # eda.LR()
    # eda.LinearSVM()
    # eda.KNN()
    # eda.gaussian()
    # eda.bernouli()
    # eda.Neural()

    eda.RF()
    eda.GBDT()