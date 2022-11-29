import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def do_Kfold(model, X, y, k, scaler=None, random_state=146):
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=k, random_state = random_state, shuffle=True)

    train_scores = []
    test_scores = []

    for idxTrain, idxTest in kf.split(X):
        Xtrain = X[idxTrain, :]
        Xtest = X[idxTest, :]
        ytrain = y[idxTrain]
        ytest = y[idxTest]

        if scaler != None:
            Xtrain = scaler.fit_transform(Xtrain)
            Xtest = scaler.transform(Xtest)

        model.fit(Xtrain,ytrain)

        train_scores.append(model.score(Xtrain,ytrain))
        test_scores.append(model.score(Xtest,ytest))
        
    return train_scores,test_scores
    
#functions courtesy of Prof. Smith
   
def plot_groups(points, groups, colors, 
               ec='black', ax=None,s=30, alpha=0.5,
               figsize=(6,6), labels = ['x','y'], legend_text = None):
        
    '''Creates a scatter plot, given:
            Input:  points (2D array)
                    groups (1D array containing an integer label for each point)
                    colors (one for each group)
                    s (size for points)
                    alpha (transparency)
                    legend_text (a list of labels for the legend)
            Output: handle to the ax object (the current axes/plot)
    '''
    
    if ax is None:
        fig,ax = plt.subplots(figsize=figsize)
    else:
        fig = plt.gcf()
    
    for yi in np.unique(groups):
        idx = groups == yi
        plt.scatter(points[idx,0], points[idx,1],color = colors[yi],
               alpha = alpha, s = s, ec = 'k')
        plt.xlabel(labels[0], fontsize = 14)
        plt.ylabel(labels[1], fontsize = 14)
        if legend_text:
            plt.legend(legend_text)
    return fig,ax

def compare_classes(actual, predicted, names=None):
    '''Function returns a confusion matrix, and overall accuracy given:
            Input:  actual - a list of actual classifications
                    predicted - a list of predicted classifications
                    names (optional) - a list of class names
    '''
    accuracy = sum(actual==predicted)/actual.shape[0]
    
    classes = pd.DataFrame(columns = ['Actual', 'Predicted'])
    classes['Actual'] = actual
    classes['Predicted'] = predicted

    conf_mat = pd.crosstab(classes['Actual'], classes['Predicted'])
    
    if type(names) != type(None):
        conf_mat.index = names
        conf_mat.index.name = 'Actual'
        conf_mat.columns = names
        conf_mat.columns.name = 'Predicted'
    
    print('Accuracy = ' + format(accuracy, '.2f'))
    return conf_mat, accuracy


def get_colors(N, map_name='rainbow'):
    '''Returns a list of N colors from a matplotlib colormap
            Input: N = number of colors, and map_name = name of a matplotlib colormap
    
            For a list of available colormaps: 
                https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html
    '''
    import matplotlib
    cmap = matplotlib.cm.get_cmap(name=map_name)
    n = np.linspace(0,1,N)
    colors = cmap(n)
    return colors