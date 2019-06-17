from sklearn import preprocessing, svm, neighbors, metrics
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, ShuffleSplit
import matplotlib.pyplot as plt
from DealtextUtil import DealtextUtil


class RegressionModel(object):
    def get_data(self):
        dtu = DealtextUtil()
        data = dtu.readdata("./data/cluster3_im.csv")
        npdata = data.values
        X = npdata
        y = npdata[:, -1]
        # 先进行归一化
        X_name = X[:, 0]
        X_topro = X[:, 1:10]
        X_rest = X[:, 10:]
        X_topro = preprocessing.scale(X_topro)
        X = np.c_[X_name, X_topro]
        X = np.c_[X, X_rest]
        return X, y

    def get_train_data(self):
        X, y = self.get_data()
        X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)
        X_test = X_test1[:, 1:10]
        X_train = X_train1[:, 1:10]
        y_train = X_train1[:, -4:-2]
        y_test = X_test1[:, -4:-2]
        return X_train, X_test, y_train, y_test

    # 主要做测试用，i为随机种子
    def get_train_data_ran(self, i):
        X, y = self.get_data()
        X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=i)
        X_test = X_test1[:, 1:10]
        X_train = X_train1[:, 1:10]
        y_train = X_train1[:, -4:-2]
        y_test = X_test1[:, -4:-2]
        return X_train, X_test, y_train, y_test

    # 评估，评估结果的好坏
    def lx_eval(self, y_pred, y):
        y_pred = np.array(y_pred).reshape(-1, 1)
        y = np.array(y).reshape(-1, 1)
        evals = np.mean(abs(y_pred - y))
        return evals

    # 用svm做刑期预测
    def SVR_model_xq(self, X_train, X_test, y_train, y_test):
        model = svm.SVR()
        model = model.fit(X_train, y_train[:, 1])
        y_pred=model.predict(X_test)
        print('刑期差额:', self.lx_eval(y_pred, y_test[:, 1]))
        return model

    # 用svm做金额预测
    def SVR_model_je(self, X_train, X_test, y_train, y_test):
        model = svm.SVR()
        model = model.fit(X_train, y_train[:, 1])
        y_pred=model.predict(X_test)
        print('金额差额：', self.lx_eval(y_pred, y_test[:, 0]))
        return model

    # knn预测金额的模型
    def knn_model_je(self,X_train, X_test, y_train, y_test):
        model = neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='auto')
        model.fit(X_train, y_train[:, 0])
        y_pred = model.predict(X_test)
        print('金额差额：', self.lx_eval(y_pred, y_test[:, 0]))
        return model

    # 回归模型，用于预测刑期
    def knn_model_xq(self, X_train, X_test, y_train, y_test):
        model = neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='ball_tree')
        model.fit(X_train, y_train[:, 1])
        y_pred = model.predict(X_test)
        print('刑期差额：', self.lx_eval(y_pred, y_test[:, 1]))
        return model

    # 交叉验证
    def cv_scores(self, model, X, y):
        kfold = KFold(n_splits=10, shuffle=True, random_state=6)
        cvs = cross_val_score(model, X, y, cv=kfold)
        print(cvs)
        score = np.mean(cvs)

        return score

    # 用于计算刑期或者罚金
    def predict_value(self, model, x_topred):
        y_pred = model.predict(x_topred)
        return y_pred

    # 规范化数据用
    def formatdata(self,yc):
        for data in yc:
            if data[0] < 2:
                data[0] = 2
            data[0] = round(data[0], 1)
            data[1] = round(data[1])
        return yc

    # 规范化数据用
    def formatdata2(self,yc):
        for data in yc:
            if data[0] < 2:
                data[0] = 2
        return yc

    # 量刑范围
    def lxfw(self, data):
        data = self.formatdata2(data)
        jine = data[:, 0]
        xingq = data[:, 1]

        jine = jine.reshape(1, 3)
        xingq = xingq.reshape(1, 3)
        jine = np.sort(jine, axis=0)
        xingq = np.sort(xingq, axis=0)
        jinestr = ''
        xinqstr = ''
        min = jine.min()
        max = jine.max()
        if (jine.min() != jine.max()):
            jinestr = '%.1f---%.1f' % (jine.min(), jine.max())
        else:
            jinestr = str(jine.min())
        if xingq.min() != xingq.max():
            xingqstr = '%d---%d' % (xingq.min(), xingq.max())
        else:
            xingqstr = str(xingq.max())
        data = np.array([jinestr, xingqstr])
        return data

    # 用于获得推荐的刑期范围和金额范围，X_train1为训练集，points为测试集
    def judge_fw(self, model, X_train1, points):
        fwdata1 = []
        for i in range(len(points)):
            a, b = model.kneighbors([points[i]])
            lxdata = X_train1[b]
            lxdata.resize(3, 15)
            lxdata = lxdata[:, -4:-2]

            fwdata = self.lxfw(lxdata)
            fwdata1.append(fwdata)

        return fwdata1

    #存储模型
    def save_model_je(self):
        X_train, X_test, y_train, y_test=self.get_train_data()
        model = self.knn_model_je(X_train, X_test, y_train, y_test)
        joblib.dump(model, './model/training_model/model_je.m')

    #存储模型
    def save_model_xq(self):
        X_train, X_test, y_train, y_test = self.get_train_data()
        model = self.knn_model_xq(X_train, X_test, y_train, y_test)
        joblib.dump(model, './model/training_model/model_xq.m')

    #载入模型
    def load_model_je(self):
        model = joblib.load('./model/model_je.m')
        return model

    #载入模型
    def load_model_xq(self):
        model = joblib.load('./model/model_xq.m')
        return model

    #将外部传入的数据和原数据做统一的标准化
    def get_prodata(self,vec):
        dtu=DealtextUtil()
        data = dtu.readdata("./data/cluster3_im.csv")
        X = data.values
        X1 = X[:, 1:10]
        y = X[:, -1]
        scaler = preprocessing.StandardScaler().fit(X1)

        vec = scaler.transform(vec)

        return X, y, vec

    # 接受外部数据并进行判断
    def get_result_outer(self,v):
        X, y, vec = self.get_prodata(v)
        X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=23)

        model_je = self.load_model_je()
        je = self.predict_value(model_je, vec)

        model_xq = self.load_model_xq()
        xq = self.predict_value(model_xq, vec)

        r_precise = np.c_[je, xq]
        r_precise = self.formatdata(r_precise)

        fw = self.judge_fw(model_xq, X_train1, vec)

        return r_precise, fw

    # 内部进行测试的时候使用
    def get_result_inner(self):
        X, y = self.get_data()
        # print(X)
        X_train1, X_test1, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
        X_test = X_test1[:, 1:10]
        X_train = X_train1[:, 1:10]
        y_train = X_train1[:, -4:-2]
        y_test = X_test1[:, -4:-2]
        model_je = self.knn_model_je(X_train, X_test, y_train, y_test)
        model_xq = self.knn_model_xq(X_train, X_test, y_train, y_test)
        y_pred_je = self.predict_value(model_je, X_test)
        y_pred_xq = self.predict_value(model_xq, X_test)
        y_pred = np.c_[y_pred_je, y_pred_xq]
        y_pred = self.formatdata(y_pred)
        # print('after_format',lx_eval(y_pred[:,0],y_test[:,0]))
        fw = self.judge_fw(model_je, X_train1, X_test)
        print('定值：',y_pred)
        print('范围:',fw)

    #调参
    def gridS(self):
        ev = metrics.make_scorer(self.lx_eval, greater_is_better=False)
        X_train, X_test, y_train, y_test = self.get_train_data()

        clf = neighbors.KNeighborsRegressor(n_neighbors=3)
        algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
        weights = ['uniform', 'distance']
        # metric=['minkowski','cosine']
        param_test = dict(algorithm=algorithm, weights=weights)
        gridKnn = GridSearchCV(clf, param_test, cv=10, scoring=ev)
        gridKnn.fit(X_train, y_train[:, 0])
        print('best score is:', str(gridKnn.best_score_))
        print('best params are', str(gridKnn.best_params_))

    # d读入外部文件，进行预测,rp为精确预测，fw返回值为范围
    def get_result_outfile(self,path):
        dtu=DealtextUtil()
        data = dtu.readdata(path)
        npdata = data.values
        npdata1 = npdata[:, 1:10]
        rp, fw = self.get_result_outer(npdata1)
        return rp, fw


    #画出学习曲线
    def draw_learning_curve(self):
        X, y = self.get_data()
        X = X[:, 1:10]
        y = X[:, -4:-2]
        model = self.load_model_je()
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)
        train_sizes, train_scores, test_scores = learning_curve(model, X=X, y=y[:, 0],
                                                                train_sizes=np.linspace(0.1, 1.0, 10), cv=cv,
                                                                scoring='neg_mean_absolute_error'
                                                                )
        print(train_sizes)
        print('----------')
        print(np.mean(train_scores, axis=1))
        print('----------')
        print(np.mean(test_scores, axis=1))
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure()
        plt.title('learning_curve')
        plt.plot(train_sizes, np.mean(train_scores, axis=1), 'r', label='train_scores')
        plt.plot(train_sizes, np.mean(test_scores, axis=1), 'b', label='test_scores')
        plt.legend(['train_scores', 'test_scores'])
        plt.show()