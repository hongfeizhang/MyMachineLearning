import csv
import numpy as np
import pandas as pd
import scipy.io as sio
import matplotlib.pyplot as plt

#from oct2py import octave
from sklearn.model_selection import GridSearchCV,PredefinedSplit
from sklearn.svm import SVC,LinearSVC


def plot_data(X,y):
    plt.plot(X[y==0,0],X[y==0,1],'yo',label='neg')
    plt.plot(X[y==1,0],X[y==1,1],'b+',label='pos')
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='lower left')
    
def plot_boundary(X,clf,level=0.0):
    x1_min,x1_max=X[:,0].min(),X[:,0].max()
    x2_min,x2_max=X[:,1].min(),X[:,1].max()
    
    h=0.01
    grid_x1,grid_x2=np.meshgrid(np.arange(x1_min,x1_max,h),
                                np.arange(x2_min,x2_max,h))
    grid_y=clf.predict(np.c_[grid_x1.ravel(),grid_x2.ravel()])
    grid_y=grid_y.reshape(grid_x1.shape)
    
    x1_extra=(x1_max-x1_min)/50
    x2_extra=(x2_max-x2_min)/50
    plt.xlim(x1_min-x1_extra,x1_max+x1_extra)
    plt.ylim(x2_min-x2_extra,x2_max+x2_extra)
    
    plt.contour(grid_x1,grid_x2,grid_y,levels=[level])
   
def plot_support_vectors(clf):
    '''Plot the support vectors from trained support vector classifier clf.
    
    Only the support vectors of the first two classes are drawn.
    '''
    
    # Obtain support vectors for first two classes
    sv_neg, sv_pos = np.vsplit(clf.support_vectors_, np.cumsum(clf.n_support_)[0:1])

    # Plot support vectors
    plt.plot(sv_neg[:,0], sv_neg[:,1], 'r+', label='neg SVs', alpha=0.5)
    plt.plot(sv_pos[:,0], sv_pos[:,1], 'rx', label='pos SVs', alpha=0.5)
    plt.legend()
    

data=sio.loadmat('ex6data3.mat')

# Training data 
X_train_0 = data['X']
y_train_0 = data['y'].ravel()

# Validation data
X_cv_0 = data['Xval']
y_cv_0 = data['yval'].ravel()    
    
    
plot_data(X_train_0, y_train_0)
plt.title('Training data')

plot_data(X_cv_0, y_cv_0)
plt.title('Validation data')

X = np.concatenate([X_train_0, X_cv_0])
y = np.concatenate([y_train_0, y_cv_0])

num_train = X_train_0.shape[0]
num_cv = X_cv_0.shape[0]

cv_fold = np.empty(num_train + num_cv, dtype='int8')
cv_fold[:num_train] = -1
cv_fold[num_train:] = 0
cv = PredefinedSplit(cv_fold)

# Values for grid search (see description in ex6.pdf)
grid = np.array([0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30])

# Grid values for C
grid_C = grid
# Grid values for gamma
grid_gamma = 1 / (grid ** 2)

# Classifier used for grid search
clf = SVC(kernel='rbf')

# Grid search to find the best C and gamma values using the predefined
# training and validation set.
gs = GridSearchCV(clf, param_grid={'C':grid_C, 'gamma':grid_gamma}, cv=cv)
gs.fit(X, y)
pd.DataFrame(gs.cv_results_)

clf_best = gs.best_estimator_
print('best C value =', clf_best.C)
print('best gamma value =', clf_best.gamma)


plot_data(X_train_0, y_train_0)
plot_boundary(X_train_0, clf_best)
plt.title(f'C={clf_best.C:.4}, gamma={clf_best.gamma:.4}')

