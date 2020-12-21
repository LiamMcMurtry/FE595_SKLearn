# Liam McMurtry
# FE595
# coding: utf-8

# In[41]:


from sklearn import linear_model, cluster, datasets
import matplotlib.pyplot as plt
import pandas as pd


def boston_linear():
    # Here, we first load in the Boston dataset
    boston = datasets.load_boston()
    # Here, we convert the boston data to a dataframe format
    boston_df = pd.DataFrame(boston.data, columns=boston.feature_names.tolist())
    # Here, we run a linear regression on the dataframe
    lm = linear_model.LinearRegression().fit(boston_df, boston.target)
    # Her, we make a dataframe that has feature names, coefficient values, and absolute values
    sorting_df = pd.DataFrame(columns=['feature', 'coefs'])
    sorting_df['feature'] = boston.feature_names.tolist()
    sorting_df['coefs'] = lm.coef_.tolist()
    sorting_df['coefs_abs'] = abs(sorting_df['coefs'])
    # Now, we print the coefficients and the feature names, which will be sorted by the absolute value column.
    # We are looking what which will have the greatest impact, not just the ones that have
    # the highest positive value, so sorting by absolute value then becomes necessary. 
    print(sorting_df.sort_values('coefs_abs', ascending = False).reset_index(drop=True)[['feature', 'coefs']])
    
def elbow_iris():
    # Here, we first load in the iris dataset
    iris = datasets.load_iris()
    # Next, we create a dataframe out of the iris data
    iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
    # Here, we define two empty lists
    ssr = []
    x = []
    # Here, we perform k-means clustering analysis, fitted to the iris dataframe, 
    # and giving a sum of squared errors for each possible k-cluster value to test which number is the best
    for k in range(1, 15):
        k_means = cluster.KMeans(n_clusters=k).fit(iris_df)
        ssr.append(k_means.inertia_)
        x.append(k)
    # After inspecting the graph produced by below, it can be seen that the best number of clusters is three,
    # which implies that there are indeed three populations here. This can confirmed in the graph 
    plt.plot(x[0:3], ssr[0:3], 'g')
    plt.plot(x[2:], ssr[2:], 'r')
    plt.plot(x[2], ssr[2], 'bo')
    plt.title("Elbow Graph")
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Error')
    plt.show()

    
if __name__ == '__main__':
    boston_linear()
    elbow_iris()


# In[ ]:




