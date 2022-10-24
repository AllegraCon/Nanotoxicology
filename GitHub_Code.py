# -*- coding: utf-8 -*-
"""
Created on 14/01/2022

@author: Allegra Conti

Paper: Predicting the cytotoxicity of nanomaterials through explainable, extreme gradient boosting 
Authors: 
Allegra Conti*, Luisa Campagnolo°, Stefano Diciotti^, Antonio Pietroiusti+, Nicola Toschi*$

*Medical Physics Section, Department of Biomedicine and Prevention, University of Rome Tor Vergata, Rome, Italy; 
°Histology and Embryology Section, Department of Biomedicine and Prevention, University of Rome Tor Vergata, Rome, Italy; 
^Department of Electrical, Electronic, and Information Engineering ‘Guglielmo Marconi’, University of Bologna, Cesena, Italy; Alma Mater Research Institute for Human-Centered Artificial Intelligence, Bologna, Italy; 
+Unicamillus University of Medical Sciences, Rome, Italy; 
$Athinoula A. Martinos Center for Biomedical Imaging and Harvard Medical School, Boston, MA, USA

"""




method='XGBoost-Regression'
test=0.3; #size of the test set 0.3==30%

ttest=str(test)
testset='testset:'+str(test) 
rm=10; ##number of random states
randomstate='random state n.: '+str(rm) ##change in line 40 when u change thin

import scipy
from scipy import stats


import pandas as pd
import numpy as np
import statistics
import matplotlib
DATALOAD = pd.read_excel (r'DATA.xlsx')

###import the necessary module
from sklearn import preprocessing
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import mean_squared_error
import shap
from bayes_opt import BayesianOptimization
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import OneHotEncoder
# create the Labelencoder object
le = preprocessing.LabelEncoder()

## remove lines with missing data
DATALOAD.reset_index(inplace = True, drop = True) 


# select columns other than target
cols = [col for col in DATALOAD.columns]
col_pred=25; ##last column with predictors
data2 = DATALOAD[cols[0:col_pred]];
cat = [cols[0:6]];
cc=col_pred  #first column with target ######################################


ii=0
while ii<col_pred:
       data2[cols[ii]] = pd.to_numeric(data2[cols[ii]]);
       ii+=1


features_in_pca=np.empty((2, 7,len(cols))); ##two is the number of Principal components, 6 n. of features on which PCA is calculated

#setting variables as categotical as for example
#data2[cols[6]] = data2[cols[6]].astype('category')

  

       #################### ****************** OneHotEncoder - multiclass variables
from sklearn.preprocessing import OneHotEncoder
# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')


#repeat lines from 86 to 94 below for all categorical variables
variable = pd.concat([data2[cols[6]],data2[cols[6]]], axis = 1) ;
variable.columns = ['Names', cols[6]]
        
enc_df = pd.DataFrame(enc.fit_transform(variable[[cols[6]]]).toarray())
dum_df = pd.get_dummies(variable, columns=[cols[6]], prefix=[cols[6]] )
df0= dum_df.drop('Names', 1)
        
del variable, enc_df, dum_df

##Concatenate all variables (old+all variables came out fromthe OneHotEncoder) as for example for 6 categorical variables


#data_all2= pd.concat([data2[cols[0]],data2[cols[1]],data2[cols[2]],data2[cols[3]],data2[cols[4]],data2[cols[5]],data2[cols[6]],
#                  data2[cols[7]], data2[cols[8]], data2[cols[9]], data2[cols[10]], data2[cols[11]],
#                  data2[cols[14]], data2[cols[16]],data2[cols[19]], data2[cols[20]], data2[cols[21]], data2[cols[22]],data2[cols[23]],
#                  data2[cols[24]], 
#                  df0, df1,df2,df3,df4,df5,df6],axis=1)



del df0
cols2 = [col for col in data_all2.columns]


while cc<len(cols): ###for each target defined by cols
    target=cols[cc];
    

    from sklearn.preprocessing import StandardScaler
    data_all = pd.concat([data_all2, DATALOAD[target]], axis = 1) ;
    
    #drop in Nan
    data_all=data_all.replace(' ', np.nan, regex=True)
    data_all.dropna(inplace=True)
    data_all.reset_index(inplace = True, drop = True) 
    
    ##removing outliers
    valid_data1=data_all[data_all[target]>data_all[target].quantile(0.10)];
    valid_data=valid_data1[valid_data1[target]<valid_data1[target].quantile(0.90)];
    del data_all
    data_all=valid_data
    vv=round(len(data_all.index)*test); 
  
    datatarget = data_all[target]; 
   
    filename=method+'test'+ttest+'.txt'
    file=method+'test'+testset

    from sklearn.model_selection import train_test_split
    
    randomst=list(range(rm))
   
    #for rs in randomst
    rs=0
    
    variable=data_all.shape
    colonne=variable[1]-1;
    del variable
    acc2= [None] * (len(randomst))
    sensitivity2=[None] * (len(randomst)) 
    specificity2=[None] * (len(randomst))
    auc2=[None] * (len(randomst))
    shap_values = np.empty((len(randomst)*(len(data_all)-int(vv)+1),colonne+1))
    shap_val= np.empty(shape=(len(randomst),len(data_all)-int(vv)+1,colonne+1))

    A=np.zeros(shape=(len(randomst)))
    Ascal=np.zeros(shape=(len(randomst)))
    slopes=np.zeros(shape=(len(randomst)))
    intercepts=np.zeros(shape=(len(randomst)))
    p_values=np.zeros(shape=(len(randomst)))
    r_values=np.zeros(shape=(len(randomst)))
    std_errs=np.zeros(shape=(len(randomst)))
    R2s=np.zeros(shape=(len(randomst)))
    coef_sp_s=np.zeros(shape=(len(randomst)))
    p_sp_s=np.zeros(shape=(len(randomst)))
    Xtestload=np.empty((len(randomst)*(len(data_all)-int(vv)),colonne+1))
    YP=np.empty((len(randomst)*(int(vv))))*np.nan;
    YT=np.empty((len(randomst)*(int(vv))))*np.nan;
    #papa=np.zeros(5,len(cols)-(col_pred))
    papa={}
    loads= np.empty(shape=(len(randomst),7,7))*np.nan
    varianza= np.empty(shape=(len(randomst),7))*np.nan
    counts_tot=np.empty(shape=(len(randomst)))*np.nan
    
    import datetime
    begin_time = datetime.datetime.now()
    
    
    
    #trasnform variables into categorical

    
    
    while rs<len(randomst):
        
         # ##Split dataset into training set and test set
        Xt, Xv, yt, yv = train_test_split(data_all.loc[:, data_all.columns != target], datatarget, test_size=test, random_state=rs,shuffle = True) # 70% training and 30% test
        
       
        
        #########################
        ##PCA

        from sklearn.decomposition import PCA
        x=Xt[cols[0:7]];
        from factor_analyzer import FactorAnalyzer
        from factor_analyzer.factor_analyzer import calculate_kmo
        #kmo_all,kmo_model=calculate_kmo(x)
        ##kmo_model
        # Create factor analysis object and perform factor analysis
        fa = FactorAnalyzer(method='principal', rotation="varimax")
        fa.fit(x)

        # Check Eigenvalues
        eigen_values, vectors = fa.get_eigenvalues()
    
        ##check for eigenvalues larger than 1
        ee=0.99999;
        count = 0
        for i in eigen_values :
            if i > ee :
                count = count + 1
    
        counts_tot[rs]=count;
        fa = FactorAnalyzer(rotation="varimax",method='principal',n_factors=count)
        fa.fit(x);
        
        
        loads[rs,:,0:count]=fa.loadings_;
        variable=fa.get_factor_variance();

        varianza[rs,0:count]=variable[2];##varianza cumulata [PC1, PC2]
        del variable
        x_transformed = fa.fit_transform(x)
        na=list(range(1, count+1));#lista di nomi colonne dataframe PCA
        nana=''.join([str(item) for item in na])
        xx_t=pd.DataFrame(x_transformed,
             columns=np.char.mod('%d', na))
        bla=Xt.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
        finalDf = pd.concat([xx_t, bla[cols2[8:len(bla.columns)]]], axis = 1) ; 
        data=    finalDf;
        del finalDf  
        del x_transformed
        del x, Xt
        Xt=data;
        Xt = Xt.astype({"Coating_Binary": int})
        
        del data
        ##trasnformation of the test set with the same trasformation used for the training set
        x=Xv[cols[0:7]];
        x_transformed = fa.fit_transform(x);
        xx_v=pd.DataFrame(x_transformed,
             columns=np.char.mod('%d', na))
        bla=Xv.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill='')
        finalDf = pd.concat([xx_v, bla[cols2[8:len(bla.columns)]]], axis = 1) ; 
        del Xv
        Xv=finalDf;
        del finalDf
        Xv = Xv.astype({"Coating_Binary": int})
        

        ##recording all different training tests
        if rs==0: 
            Xtestload=Xt;
        else: 
            Xtestload=np.append(Xtestload,Xt, axis=0)
    
 
        cv_params = {'max_depth': [1,2,3,4,5,6], 'min_child_weight': [1,2,3,4],'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]}    # parameters to be tries in the grid search
        fix_params = {'n_estimators': 100, 'objective': 'reg:squarederror'}   #other parameters, fixed for the moment 
      
       
        clf = xgb.XGBRegressor(
            tree_method="gpu_hist", enable_categorical=True )
        
        csv = GridSearchCV(clf, cv_params, scoring = 'neg_root_mean_squared_error', cv = 5)
        csv.fit(Xt, yt) #runs the search

        params=  csv.best_params_;
      

    
    
        #Now we train our final model on the entire training set, and evaluate it on the still unused testing set:

        #gbm = xgb.XGBRegressor(params)
        gbm=xgb.XGBRegressor(**params, objective='reg:squarederror', eval_metric='rmse')
        
        gbm.fit(Xt,yt)
        gbm.score(Xt,yt)
        gbm.score(Xv,yv)
        y_pred = gbm.predict(Xv)
        
        yvv=yv.reset_index(level=None, drop=False, inplace=False)
        columns=['B', 'C']
        if rs==0: 
            YT=yvv[target];
            YP=y_pred;
        else: 
            YT=np.append(YT,yvv[target], axis=0)
            YP=np.append(YP,y_pred, axis=0)
        
        del yvv
        
    
    #Import scikit-learn metrics module for accuracy calculation
        from sklearn import metrics
        from sklearn.metrics import roc_curve, auc,recall_score,precision_score


        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import plot_confusion_matrix
        import matplotlib.pyplot as pl

    
        A[rs]=np.sqrt(mean_squared_error(yv, y_pred)) #rmse
        tt=shap.TreeExplainer(gbm).shap_values(Xt)
        #cc = shap.TreeExplainer(model).shap_values(Xv)
        if rs==0: 
            shap_val=tt;
        else: 
            shap_val=np.append(shap_val,tt, axis=0)
    
        
       
        #shap_val_in[rs,:,:]=cc
        slopes[rs], intercepts[rs], r_values[rs], p_values[rs], std_errs[rs] = scipy.stats.linregress(yv, y_pred)
        R2s[rs]=r2_score(yv, y_pred);
        coef_sp_s[rs], p_sp_s[rs] = spearmanr(yv, y_pred)
        
        ##delete nan
        nan_array1 = np.isnan(p_sp_s)
        not_nan_array1 = ~ nan_array1
        array2 = (p_sp_s[not_nan_array1])
        
        nan_array2 = np.isnan(coef_sp_s)
        not_nan_array2 = ~ nan_array2
        array3 = (coef_sp_s[not_nan_array2])
    
  
        papa[rs]=params;
        
        
        rs +=1
        

    teatime=(datetime.datetime.now() - begin_time)  
    # %%B slope, C intercept, D r_value, p_value,
    import os
    path = (r'D:\Folder_Output')
    os.chdir(path)
    myfile = open(filename, 'a')

# Write performances on a file (median and mad of Spearman's correlation coefficient and p-value)
    myfile.write('\n \n+ ***'+target+method+testset+'n.Random States.'+ str(len(randomst))+'\n r_value:'+ str(np.median(r_values))+
                 '\n coef-Spearman:'+str(np.median(array3))+'\n coef-Spearman MAD:'+str(scipy.stats.median_absolute_deviation(array3))+
                 '\n p-Spearman:'+str(np.median(array2))+'\n p-Spearman MAD:'+str(scipy.stats.median_absolute_deviation(array2))+
                 '\n Execution time:'+str(teatime)
                 )
    
   

# Join various path components  
    pippo=os.path.join(path, target, "")
    os.mkdir(pippo)
    os.chdir(pippo)

    ##Output shap
    import pandas as pd 
    df = pd.DataFrame(xplot, columns = Xt.columns)

    df.to_excel (r'output_x.xlsx', index = False, header=True)

    df = pd.DataFrame(shap_val, columns = Xt.columns)

    df.to_excel (r'output_shap.xlsx', index = False, header=True)
    
    df = pd.DataFrame(np.median(loads, axis=0))

    
    
  
    
    
    os.chdir(path)
    cc+=1
    
    if cc==len(cols)+1:
        myfile.close()
