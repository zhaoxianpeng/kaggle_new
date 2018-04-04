# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       xpzhao
   date：          18-4-4
-------------------------------------------------
   Change Activity:
                   18-4-4:
-------------------------------------------------
"""
__author__ = 'xpzhao'

from titanic import feature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier


def get_out_fold(clf, x_train, y_train, x_test):
    from sklearn.model_selection import KFold
    ntrain = x_train.shape[0]
    ntest = x_test.shape[0]
    oof_train = np.zeros((ntrain,))
    oof_test = np.zeros((ntest,))
    SEED = 0  # for reproducibility
    NFOLDS = 7  # set folds for out-of-fold prediction
    kf = KFold(n_splits=NFOLDS, random_state=SEED, shuffle=False)
    oof_test_skf = np.empty((NFOLDS, ntest))

    for i, (train_index, test_index) in enumerate(kf.split(x_train)):
        x_tr = x_train[train_index]
        y_tr = y_train[train_index]
        x_te = x_train[test_index]

        clf.fit(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


def test_main():
    titanic_train_data_X, titanic_train_data_Y, titanic_test_data_X, PassengerId = feature.feature_eng()
    # 模型融合及测试
    #
    # 模型融合的过程需要分几步来进行。
    #
    # (1) 利用不同的模型来对特征进行筛选，选出较为重要的特征：

    # (2) 依据我们筛选出的特征构建训练集和测试集
    #
    # 但如果在进行特征工程的过程中，产生了大量的特征，
    # 而特征与特征之间会存在一定的相关性。
    # 太多的特征一方面会影响模型训练的速度，另一方面也可能会使得模型过拟合。
    # 所以在特征太多的情况下，我们可以利用不同的模型对特征进行筛选，选取出我们想要的前n个特征。
    feature_to_pick = 30
    feature_top_n, feature_importance = feature.get_top_n_features(titanic_train_data_X, titanic_train_data_Y, feature_to_pick)
    titanic_train_data_X = pd.DataFrame(titanic_train_data_X[feature_top_n])
    titanic_test_data_X = pd.DataFrame(titanic_test_data_X[feature_top_n])
    feature.show_feature_select(feature_importance)

    # (3) 模型融合（Model Ensemble）
    #
    # 常见的模型融合方法有：Bagging、Boosting、Stacking、Blending。
    # (3-1):Bagging
    #
    # Bagging 将多个模型，也就是多个基学习器的预测结果进行简单的加权平均或者投票。它的好处是可以并行地训练基学习器。Random Forest就用到了Bagging的思想。
    #
    # (3-2): Boosting
    #
    # Boosting 的思想有点像知错能改，每个基学习器是在上一个基学习器学习的基础上，对上一个基学习器的错误进行弥补。我们将会用到的 AdaBoost，Gradient Boost 就用到了这种思想。
    #
    # (3-3): Stacking
    #
    # Stacking是用新的次学习器去学习如何组合上一层的基学习器。如果把 Bagging 看作是多个基分类器的线性组合，那么Stacking就是多个基分类器的非线性组合。Stacking可以将学习器一层一层地堆砌起来，形成一个网状的结构。
    #
    # 相比来说Stacking的融合框架相对前面的二者来说在精度上确实有一定的提升，所以在下面的模型融合上，我们也使用Stacking方法。
    #
    # (3-4): Blending
    #
    # Blending 和 Stacking 很相似，但同时它可以防止信息泄露的问题。

    # Stacking框架融合:
    #
    # 这里我们使用了两层的模型融合，Level 1使用了：RandomForest、AdaBoost、ExtraTrees、GBDT、DecisionTree、KNN、SVM ，一共7个模型，
    # Level 2使用了XGBoost使用第一层预测的结果作为特征对最终的结果进行预测。
    #
    # Level 1：
    #
    # Stacking框架是堆叠使用基础分类器的预测作为对二级模型的训练的输入。
    #  然而，我们不能简单地在全部训练数据上训练基本模型，产生预测，输出用于第二层的训练。
    # 如果我们在Train Data上训练，然后在Train Data上预测，就会造成标签泄露。
    # 为了避免标签泄露，我们需要对每个基学习器使用K-fold，将K个模型对Valid Set的预测结果拼起来，作为下一层学习器的输入。
    #
    # 所以这里我们建立输出K-fold预测的方法 get_out_fold

    # 构建不同的基学习器，这里我们使用了RandomForest、AdaBoost、ExtraTrees、GBDT、DecisionTree、KNN、SVM 七个基学习器：
    # （这里的模型可以使用如上面的GridSearch方法对模型的超参数进行搜索选择）
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC

    rf = RandomForestClassifier(n_estimators=500, warm_start=True, max_features='sqrt', max_depth=6,
                                min_samples_split=3, min_samples_leaf=2, n_jobs=-1, verbose=0)

    ada = AdaBoostClassifier(n_estimators=500, learning_rate=0.1)

    et = ExtraTreesClassifier(n_estimators=500, n_jobs=-1, max_depth=8, min_samples_leaf=2, verbose=0)

    gb = GradientBoostingClassifier(n_estimators=500, learning_rate=0.008, min_samples_split=3, min_samples_leaf=2,
                                    max_depth=5, verbose=0)

    dt = DecisionTreeClassifier(max_depth=8)

    knn = KNeighborsClassifier(n_neighbors=2)

    svm = SVC(kernel='linear', C=0.025)
    # 将pandas转换为arrays：
    # Create Numpy arrays of train, test and target (Survived) dataframes to feed into our models
    x_train = titanic_train_data_X.values  # Creates an array of the train data
    x_test = titanic_test_data_X.values  # Creats an array of the test data
    y_train = titanic_train_data_Y.values

    # Create our OOF train and test predictions. These base results will be used as new features
    rf_oof_train, rf_oof_test = get_out_fold(rf, x_train, y_train, x_test)  # Random Forest
    ada_oof_train, ada_oof_test = get_out_fold(ada, x_train, y_train, x_test)  # AdaBoost
    et_oof_train, et_oof_test = get_out_fold(et, x_train, y_train, x_test)  # Extra Trees
    gb_oof_train, gb_oof_test = get_out_fold(gb, x_train, y_train, x_test)  # Gradient Boost
    dt_oof_train, dt_oof_test = get_out_fold(dt, x_train, y_train, x_test)  # Decision Tree
    knn_oof_train, knn_oof_test = get_out_fold(knn, x_train, y_train, x_test)  # KNeighbors
    svm_oof_train, svm_oof_test = get_out_fold(svm, x_train, y_train, x_test)  # Support Vector

    print("Training is complete")

    # (4) 预测并生成提交文件

    # Level 2：
    #
    # 我们利用XGBoost，使用第一层预测的结果作为特征对最终的结果进行预测。
    x_train = np.concatenate(
        (rf_oof_train, ada_oof_train, et_oof_train, gb_oof_train, dt_oof_train, knn_oof_train, svm_oof_train), axis=1)
    x_test = np.concatenate(
        (rf_oof_test, ada_oof_test, et_oof_test, gb_oof_test, dt_oof_test, knn_oof_test, svm_oof_test), axis=1)

    from xgboost import XGBClassifier

    gbm = XGBClassifier(n_estimators=2000, max_depth=4, min_child_weight=2, gamma=0.9, subsample=0.8,
                        colsample_bytree=0.8, objective='binary:logistic', nthread=-1, scale_pos_weight=1).fit(x_train,
                                                                                                               y_train)
    predictions = gbm.predict(x_test)

    StackingSubmission = pd.DataFrame({'PassengerId': PassengerId, 'Survived': predictions})
    StackingSubmission.to_csv('StackingSubmission.csv', index=False, sep=',')

    # 7. 验证：学习曲线
    #
    # 在我们对数据不断地进行特征工程，产生的特征越来越多，用大量的特征对模型进行训练，
    # 会使我们的训练集拟合得越来越好，但同时也可能会逐渐丧失泛化能力，从而在测试数据上表现不佳，发生过拟合现象。
    #
    # 当然我们建立的模型可能不仅在预测集上表型不好，也很可能是因为在训练集上的表现就不佳，处于欠拟合状态。
    # 上面红线代表test error（Cross-validation error），蓝线代表train error。这里我们也可以把错误率替换为准确率，那么相应曲线的走向就应该是上下颠倒的，（score = 1 - error）。
    #
    # 注意我们的图中是error曲线。
    #
    #     左上角是最优情况，随着样本的增加，train error虽然有一定的增加吗，但是 test error却有很明显的降低；
    #     右上角是最差情况，train error很大，模型并没有从特征中学习到什么，导致test error非常大，模型几乎无法预测数据，需要去寻找数据本身和训练阶段的原因；
    #     左下角是high variance的情况，train error虽然较低，但是模型产生了过拟合，缺乏泛化能力，导致test error很高；
    #     右下角是high bias的情况，train error很高，这时需要去调整模型的参数，减小train error。
    #
    # 所以我们通过学习曲线观察模型处于什么样的状态。从而决定对模型进行如何的操作。
    # 当然，我们把验证放到最后，并不是是这一步是在最后去做。
    # 对于我们的Stacking框架中第一层的各个基学习器我们都应该对其学习曲线进行观察，
    # 从而去更好地调节超参数，进而得到更好的最终结果。
    #
    # 构建绘制学习曲线的函数：plot_learning_curve

    # 逐一观察不同模型的学习曲线：
    X = x_train
    Y = y_train

    # RandomForest
    rf_parameters = {'n_jobs': -1, 'n_estimators': 500, 'warm_start': True, 'max_depth': 6, 'min_samples_leaf': 2,
                     'max_features': 'sqrt', 'verbose': 0}

    # AdaBoost
    ada_parameters = {'n_estimators': 500, 'learning_rate': 0.1}

    # ExtraTrees
    et_parameters = {'n_jobs': -1, 'n_estimators': 500, 'max_depth': 8, 'min_samples_leaf': 2, 'verbose': 0}

    # GradientBoosting
    gb_parameters = {'n_estimators': 500, 'max_depth': 5, 'min_samples_leaf': 2, 'verbose': 0}

    # DecisionTree
    dt_parameters = {'max_depth': 8}

    # KNeighbors
    knn_parameters = {'n_neighbors': 2}

    # SVM
    svm_parameters = {'kernel': 'linear', 'C': 0.025}

    # XGB
    gbm_parameters = {'n_estimators': 2000, 'max_depth': 4, 'min_child_weight': 2, 'gamma': 0.9, 'subsample': 0.8,
                      'colsample_bytree': 0.8, 'objective': 'binary:logistic', 'nthread': -1, 'scale_pos_weight': 1}

    title = "Learning Curves"
    plot_learning_curve(RandomForestClassifier(**rf_parameters), title, X, Y, cv=None, n_jobs=4,
                        train_sizes=[50, 100, 150, 200, 250, 350, 400, 450, 500])
    plt.show()
    # 由上面的分析我们可以看出，对于RandomForest的模型，这里是存在一定的问题的，所以我们需要去调整模型的超参数，从而达到更好的效果。

    # 8. 超参数调试
    #
    # 将生成的提交文件到Kaggle提交，得分结果：
    #
    #     xgboost stacking：0.78468；
    #     voting bagging ：0.79904；
    #
    # 这也说明了我们的stacking模型还有很大的改进空间。所以我们可以在以下几个方面进行改进，提高模型预测的精度：
    #
    #     特征工程：寻找更好的特征、删去影响较大的冗余特征；
    #     模型超参数调试：改进欠拟合或者过拟合的状态；
    #     改进模型框架：对于stacking框架的各层模型进行更好的选择；
    #
    # 调参的过程.............................................慢慢尝试吧。


from sklearn.learning_curve import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), verbose=0):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt




