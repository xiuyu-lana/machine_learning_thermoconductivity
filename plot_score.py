import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


# scores = pd.read_csv('score_all.csv')
scores = pd.read_csv('score_all_p1to15.csv')
score_cv_RF = scores.values[:,1]
score_cv_AB = scores.values[:,2]
score_cv_GB = scores.values[:,3]
score_cv_V = scores.values[:,4]
model_name = scores.columns[:]
xname = scores.values[:,0]
xname = xname.tolist()
# xname = str(xname)

# plt.figure(1)
# plt.plot(score_cv_RF, '.-')
# plt.xticks(list(range(0,9)),xname)
# # plt.savefig('score_cv_RF.png')
# plt.savefig('score_cv_RF_p.png')
#
# plt.figure(2)
# plt.plot(score_cv_AB, '.-')
# plt.xticks(list(range(0,9)),xname)
# # plt.savefig('score_cv_AB.png')
# plt.savefig('score_cv_AB_p.png')
#
# plt.figure(3)
# plt.plot(score_cv_GB, '.-')
# plt.xticks(list(range(0,9)),xname)
# # plt.savefig('score_cv_GB.png')
# plt.savefig('score_cv_GB_p.png')
#
# plt.figure(4)
# plt.plot(score_cv_V, '.-')
# plt.xticks(list(range(0,9)),xname)
# # plt.savefig('score_cv_V.png')
# plt.savefig('score_cv_V_p.png')

plt.figure(5)
plt.plot(score_cv_RF, '.-', label = 'RandomForestRegressor')
plt.plot(score_cv_AB, '.-', label = 'AdaBoostRegressor')
plt.plot(score_cv_GB, '.-', label = 'GradientBoostingRegressor')
plt.plot(score_cv_V, '.-', label = 'VotingRegressor')
plt.xticks(list(range(0,9)),xname)
plt.legend()
# plt.savefig('score_all.png')
plt.savefig('score_all_p1to15.png')

# plt.show()

# print(scores)