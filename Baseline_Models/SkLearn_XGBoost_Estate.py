import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold
import skopt
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import BayesSearchCV
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import r2_score
from mpi4py import MPI


# import warnings filter
import warnings
# ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

df_1 = pd.read_csv('EState_train_0.65.csv', sep = ',', dtype={'name': str})
df_1 = df_1.dropna(axis=1)
X_train = df_1.drop('pKa', axis=1)
X_train = X_train.drop('name', axis=1)
Y_train = df_1.pKa
print(X_train.shape, Y_train.shape)

df_2 = pd.read_csv('Estate_test.csv', sep = ',', dtype={'name': str})
df_2 = df_2.dropna(axis=1)
X_test = df_2.drop('pKa', axis=1)
X_test = X_test.drop('name', axis=1)
Y_test = df_2.pKa
print(X_test.shape, Y_test.shape)

n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1200, num = 12)]

learning_rates =  [float(x) for x in np.logspace(-6, -1, num=6)]

subsample = [1.0, 0.8, 0.6, 0.4] 

max_depth = [int(x) for x in np.linspace(3, 30, num = 10)]

min_samples_split = [2, 5, 10, 15]

min_samples_leaf = [1, 2, 4, 6, 8]

max_features = [None, 'sqrt', 'log2']

search_spaces = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'subsample': subsample,
               'learning_rate': learning_rates}

scoring = {'MAE': 'neg_mean_absolute_error',
           'RMSE': 'neg_root_mean_squared_error',
           'score': 'r2'}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

bayes_cv_tuner = BayesSearchCV(estimator = GradientBoostingRegressor(),
                               search_spaces = search_spaces,
                               cv = cv,
                               n_jobs=-1,
                               verbose=0,
                               random_state=42,
                               n_iter=100,
                               scoring=scoring,
                               refit='score')

bayes_cv_tuner.fit(X_train, Y_train)

df = pd.DataFrame(bayes_cv_tuner.cv_results_)
df.to_csv('xgboost_Estate_cv_results_.csv')
if rank == 0:
    print("Best parameters set found on development set:  ")
    print()
    print(bayes_cv_tuner.best_params_)
    print()
    print("Scores on development set:")
    print()
means = bayes_cv_tuner.cv_results_['mean_test_score']
stds = bayes_cv_tuner.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, bayes_cv_tuner.cv_results_['params']):
    if rank == 0:
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

r2_train = bayes_cv_tuner.score(X_train, Y_train)
r2_test = bayes_cv_tuner.score(X_test, Y_test)
if rank == 0:
    print('r2 train, r2 test: ', r2_train, r2_test)

Y_pred = bayes_cv_tuner.predict(X_test)
sns.set(color_codes=True)
sns.set_style("white")

ax = sns.regplot(x=Y_test, y=Y_pred, scatter_kws={'alpha':0.4})
sns.regplot(x=[-15.0,31.0], y=[-15.0,31.0], scatter_kws={'alpha':0})
ax.set_xlabel('Experimental pKa', fontsize='large', fontweight='bold')
ax.set_ylabel('Predicted pKa', fontsize='large', fontweight='bold')
ax.set_xlim(round(min(Y_test))-1, round(max(Y_test))+1)
ax.set_ylim(round(min(Y_test))-1, round(max(Y_test))+1)
ax.figure.set_size_inches(5, 5)
plt.savefig('Estate_XGBoost.pdf')

if rank == 0:
    print('Test Scores:')
    print('MAE: ', mean_absolute_error(Y_test, Y_pred))
    print('RMSE: ', mean_squared_error(Y_test, Y_pred))
    print('R2: ', r2_score(Y_test, Y_pred))

Y_train_pred = bayes_cv_tuner.predict(X_train)
if rank == 0:
    print('Train Scores:')
    print('MAE: ', mean_absolute_error(Y_train, Y_train_pred))
    print('RMSE: ', mean_squared_error(Y_train, Y_train_pred))
    print('R2: ', r2_score(Y_train, Y_train_pred))
