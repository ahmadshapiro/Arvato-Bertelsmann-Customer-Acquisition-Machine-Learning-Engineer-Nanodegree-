def summarize_columns(cols,df,only_uncasted=False,samples=5,return_nan_df=False):
    '''
    Function for summarizing a column , prints the propotion if NaNs int 
    Column name 
    Sample from its values
    Values which cannot be casted into float 
    Paramters :
    col -> column 
    df -> dataset
    only_uncasted : boolean if True summarize columns which only has values that cannot be casted to float
    sample -> int to sample from values default 5
    '''
    import random
    nan_dict={}
    for column in cols:
        col = df[column]
        values_cannot_be_parsed=set()
        for value in col.value_counts().index:
            try : 
                float(value)
            except:
                values_cannot_be_parsed.add(value)
        values_cannot_be_parsed=list(values_cannot_be_parsed)
        if only_uncasted:
            if len(values_cannot_be_parsed) != 0:
                print("Column {}".format(col.name))
                if len(values_cannot_be_parsed) > 10:
                    print("Values that cannot be parsed {}".format(values_cannot_be_parsed[0:10]))
                else:
                    print("Values that cannot be parsed {}".format(values_cannot_be_parsed))
                nans=col.isna().sum()
                print("NaNs = {}".format(nans))
                print("Sample of values = {}".format(col[~col.isna()].sample(samples).values))
                print("Propotion  of NaNs = {}".format(nans/len(col)))
                print("==========================================")
        else :
            print("Column {}".format(col.name))
            if len(values_cannot_be_parsed) > 10:
                print("Values that cannot be parsed to Float {}".format(values_cannot_be_parsed[0:10]))
            else:
                print("Values that cannot be parsed to Float {}".format(values_cannot_be_parsed))
            nans=col.isna().sum()
            print("NaNs = {}".format(nans))
            values_list=list(col[~col.isna()].value_counts().index)
            if samples > len(values_list):
                samples=len(values_list)
            print("Sample of values = {}".format(random.sample(values_list,samples)))
            print("Propotion  of NaNs = {}".format(nans/len(col)))
            print("==========================================")
            nan_dict[column]=nans/len(col)
            
    if return_nan_df:
        
        return pd.DataFrame.from_dict(nan_dict,orient='index',columns=['Nan_prop'])






def save_list(l,name):
    import pickle as pk
    pk.dump(l,open("{}.data".format(name),"wb"))

def load_list(name):
    import pickle as pk
    return pk.load(open("{}.data".format(name),"rb"))

def save_pca_object(pca):
    import pickle as pk
    pk.dump(pca, open("pca_95.pkl","wb"))

def load_pca_object():
    import pickle as pk
    return pk.load(open("pca_95.pkl",'rb'))

def string_columns(df):
    cols=[]
    for col in df.columns:
        for value in df[col].values:
            try:
                float(value)
            except:
                cols.append(col)
                break
    return cols



def transform_df(df,pca_=True,scale__=True,scaler_obj=None,one_hot=True):
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    import pickle as pk
    wanted_cols=load_list("non_string_cols")
    string_cols=load_list("one_hot_cols")
    different_cols=set(df.columns).difference(set(wanted_cols).union(set(string_cols)))
    print("There are {} additional columns in the data set".format(len(different_cols)))
    ans=input("Do you wish to continue ? (y/n) (aditional columns will remain the same in the resulting data_frame)")
    if ans == "y":
        if one_hot:
            
            new_df=pd.concat([df[wanted_cols],pd.get_dummies(df[string_cols])],axis=1)
        else :
            new_df=pd.concat([df[wanted_cols],df[string_cols]],axis=1)
        if scale__ == True:
            if scaler_obj == None:
                
                scaler=MinMaxScaler()
            else:
                scaler=scaler_obj
                new_df=pd.DataFrame(data=scaler.fit_transform(new_df),columns=new_df.columns)
        if pca_ == True:
            pca=load_pca_object()
            pca_col_names=load_list("pca_col_names")
            new_df=pd.DataFrame(data=pca.transform(new_df.values),columns=pca_col_names)
        if len(different_cols) >0:
            print("Additional columns")
            for col in different_cols:
                summarize_column(df[col],samples=5)
            ans__=input("Enter the name of columns you wish to keep , separated by -> ex : col1,col2,col3 or 0 for none or all")
            if ans__ == "all":
                new_df=pd.concat([new_df,df[different_cols]],axis=1)
            elif ans__ != "0":
                if len(ans__) >1 and type(ans__)=="list" :
                    ans__=ans__.spit(',')
                    bo=all(item in list(df.columns) for item in ans__)
                else:
                    bo=ans__ in df.columns
                if bo :
                    new_df=pd.concat([new_df,df[ans__]],axis=1)
                else:
                    print("Wrong column names")
                    print("Terminating..")
                    return
        if scale__==True:
            new_df=pd.DataFrame(data=scaler.fit_transform(new_df.values),columns=new_df.columns)
        ans_2=input("Do you wish to save the new dataframe? (y/n)")
        if ans_2 == "y":
            name=input("Name (without .csv) : ")
            new_df.to_csv("{}.csv".format(name),index=False)
        return new_df




def gap_for_k(data_shape,k_inertia,k,nrefs=3):

    import numpy as np
    from sklearn.cluster import KMeans

    refrence_disp=np.zeros(nrefs)

    for i in range(nrefs):

        rand_ref=np.random.random_sample(size=data_shape)

        km=KMeans(k)
        km.fit(rand_ref)

        ref_d=km.inertia_
        refrence_disp[i]=ref_d


    gap=np.log(np.mean(refrence_disp)) - np.log(k_inertia)

    return gap


def prepare_training(scale_cluster= True,test_size=0.2,split=True,return_x_y=False,k_means=True,pca_=True,return_frame=False,one_hot=True):


    import pandas as pd
    import numpy as np
    import sklearn as sk
    import pickle as pk
    

    '''
    
    Function for preparing and splitting the training data_set
    
    Parameters :- 
                  train = csv of training data set 
                  
                  scale_cluster = scale the cluster column after k-means clustering , default = True
                  
                  test_size = Perecentage of test split. 
                  
                  split = return splitted data or not , default = True
                  
                  return_x_y = return values without concatinating  , default = False
                  
                  k_means = do Kmeans clustering , default = True
                  
                  pca_ = Do PCA , default = True
                  
                  return_Frame = return X_train and X_test as a dataframe or numpy arrays , default = False
                  
                  

    '''



    
    if pca_:
        
        x=pd.read_csv("train_pca.csv")
    elif one_hot :
        x=pd.read_csv("train_wo_pca.csv")
    else:
        x=pd.read_csv("train_wo_pca_wo_onehot.csv")
        
    y=x["RESPONSE"].values.astype(int)
    x.drop(columns="RESPONSE",inplace=True)
    
    if k_means == True:
        
    
        from sklearn.cluster import KMeans
    
        km=pk.load(open("kmean.pkl","rb"))
    
        x['cluster']=km.predict(x)
    
        if scale_cluster == True:
            
            maximum=x.cluster.max()
            x.loc[:,'cluster'] = x['cluster']/maximum


    if split:
        

        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=0)

        for train_index, test_index in sss.split(x, y):
            X_train, X_test = x.iloc[train_index,:], x.iloc[test_index,:]
            y_train, y_test = y[train_index], y[test_index]
            
        if return_frame:
            
            return X_train,X_test,y_train,y_test
        
        else:
            
            return X_train.values,X_test.values,y_train,y_test

    if return_x_y :
        
        return (x ,y) 
    
    else:
        
        return pd.concat([x,y],axis=1)
    
    
    
def get_train_xy(scale_cluster=False,values_only=False):
    
    import pandas as pd
    import sklearn as sk
    import pickle as pk
    import numpy as np 
    
    
    x=pd.read_csv("train_pca.csv")
    y=x["RESPONSE"]
    
    from sklearn.cluster import KMeans
    
    km=pk.load(open("kmean.pkl","rb"))
    
    x['cluster']=km.predict(x)
    
    
    if scale_cluster == True:
        
        maximum=x.cluster.max()
        x.loc[:,'cluster'] = x['cluster']/maximum
        
        
    if values_only:
        
        return x.values.astype(np.float32),y.values.astype(np.float32)
    
    else:
        
        return x, y 
        
def over_sample_train(over_estimator,test_size=0.2,scale_cluster=True,under_estimator=None):
    '''
    Function to over sample the training data 
    
    Pramaters :
    
        over_ratio : over sampling ratio
        k : knn for over estimator 
        under_ratio (default = None) : under sampling ratio
        split (default = False) : to return splitted sets
        test_size : test data size if splitted 
        over_estimator : object of the over sampler
        
        
    Returns :
        if split :
            X_train , X_test , y_train , y_test (numpy arrays)
        else :
            X , y (numpy arrays)
            
    '''
    import pandas as pd 
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.decomposition import PCA
    import pickle as pk
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split
    
    wanted_cols=load_list("non_string_cols")
    string_cols=load_list("one_hot_cols")
    
    train=pd.read_csv("train_clean.csv")
    
    y=train['RESPONSE'].values.astype(int)
    
    new_df=pd.concat([train[wanted_cols],pd.get_dummies(train[string_cols])],axis=1)
    
    X_train , X_test, y_train , y_test = train_test_split(new_df, y ,test_size=test_size, stratify=y)
    
    scaler=MinMaxScaler()
    X_train=pd.DataFrame(data=scaler.fit_transform(X_train),columns=X_train.columns)
    
    X_test=pd.DataFrame(data=scaler.fit_transform(X_test),columns=X_test.columns)
    
    X_train , y_train = over_estimator.fit_resample(X_train,y_train)
    
    if under_estimator != None:
        
        X_train , y_train = under_estimator.fit_resample(X_train,y_train)
    
    pca=load_pca_object()
    
    pca_col_names=load_list("pca_col_names")
    
    X_train=pd.DataFrame(data=pca.transform(X_train.values),columns=pca_col_names)
    X_test=pd.DataFrame(data=pca.transform(X_test.values),columns=pca_col_names)
    
    
    from sklearn.cluster import KMeans
    
    km=pk.load(open("kmean.pkl","rb"))
    
    
    X_train['cluster']=km.predict(X_train)
    X_test['cluster']=km.predict(X_test)
    
    
    if scale_cluster == True:
        
        maximum=X_train.cluster.max()
        X_train.loc[:,'cluster'] = X_train['cluster']/maximum
        X_test.loc[:,'cluster'] = X_test['cluster']/maximum
        
        
        
       

    return X_train.values,X_test.values,y_train,y_test
    
    
def plot_cluster(cluster_num,top_cluster_comp=10,top_pca_comp=10,plot_num=0):
    '''
    Function for plotting cluster top components 
    cluster_num = cluster number ranges from 1 to 17 
    top_cluster_comp = top highest centroids in the cluster 1 to 233
    top_pca_comp = top PCA component for each cluster centroid 1 to 399
    component
    '''
    import pickle as pk
    import pandas as pd 
    import seaborn as sns
    import matplotlib.pyplot as plt
    pca_obj=load_pca_object()
    kmeans_obj=pk.load(open("kmean.pkl","rb"))

    centroid_df=pd.DataFrame(data=kmeans_obj.cluster_centers_,columns=["Comp_{}".format(x) for x in range(1,234)],index=range(1,18))
    cluster_top_comps=[int(x)-1 for x in centroid_df.loc[cluster_num,:].sort_values(ascending=False)[:top_cluster_comp].index.str.split("_").str[1]]
    col_names=load_list("azdias_cols")
    pca_comp=pca_obj.components_
    weights_df=pd.DataFrame(data=pca_comp[cluster_top_comps],columns=col_names,index=cluster_top_comps)
    plt.figure(figsize=(10*(top_cluster_comp+1),3*(top_cluster_comp+1)))
    for i in range(1,top_cluster_comp+1):
        plt.subplot(np.ceil(top_cluster_comp/2),2,i)
        comp=cluster_top_comps[i-1]
        df_slice=(weights_df.loc[comp,:]).reindex(weights_df.loc[comp,:].abs().sort_values(ascending=False).index).to_frame(name="Weight").reset_index().rename(columns={"index":"Columns"}).iloc[:top_pca_comp,:]
        ax=sns.barplot(data=df_slice,y="Columns",x="Weight",palette="Blues_d");
        ax.set_title("PCA Component Makeup, Component #{} ,#{} Highest Component for Cluster {}".format(comp+1,i,cluster_num));


        
        
    
    
    
    
    
    
    
    
        
    
        
    
    
    
    
            
            
            
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
