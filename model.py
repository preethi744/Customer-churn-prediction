#!/usr/bin/env python
# coding: utf-8

# BANK CUSTOMER CHURN PREDICTION

# Introduction

# The customer churn prediction is about the loss of customers in a bank.We have the customer details i.e.,
#  customer_id(Id of the customer)
#  vintage(Vintage of the customer with the bank in number of days)
#  age(Age of the customer)
#  gender(Gender of the customer)
#  dependents(No:of dependents for the customer)
#  occupation(occupation of customer)
#  city(city of customer:code is given)
#  customer_nw_category(net worth of customer)
#  branch_code(code of account's branch)
#  days_since_last_transaction(no:of days since the customer's last transaction)
#  current_balance(present existing balance in account)
#  previous_month_end_balance(previous month's end day balance)
#  average_monthly_balance_prevQ(average balance in previous quarter(3-month period))
#  average_monthly_balance_prevQ2(avarage balance in previous of pervious quarter)
#  current_month_credit(total credit in current month)
#  previous_month_credit(total credit in previous month)
#  current_month_debit(total debit in current month)
#  previous_month_debit(total debit in previous month)
#  current_month_balance(this month balance)
#  previous_month_balance(previous month balance)
#  churn(0-chance of not leaving,1-chance of leaving)

# Loading the data

# In[4]:


#importing libraries for loading the data
import numpy as np
import pandas as pd
import warnings#to ignore warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


# In[7]:


df=pd.read_csv('churn_prediction.csv')


# In[8]:


df.head(10)


# In[9]:


df.shape


# In[10]:


df.info()


# DATA VISUALIZATION

# The relation between the features can be known below.

# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[12]:


df.drop(['customer_id'],axis=1,inplace=True)#dropping unwanted columns


# In[13]:


sns.countplot('churn', data=df)
plt.show()


# From above,the no:of customers leaving are less. Next,the relation between age and churn can be seen.

# In[14]:


g = sns.FacetGrid(df, col='churn')
g.map(plt.hist, 'age', bins=10)
plt.show()


# In[15]:


sns.catplot(x ='dependents', hue ='churn',kind ="count", data = df)
plt.show()


# In[16]:


sns.catplot(x ="gender", hue ='churn',kind ="count", data = df)
plt.show()


# From above plot,male customers are more who are willing to leave compared to female customers.

# In[18]:


sns.violinplot(x ='gender', y ='age', hue ='churn', data = df, split = True)
plt.show()


# From above plot we can see that the age range is between 20-40 whose churn is 1.

# In[19]:


sns.countplot(x=df.occupation)#countplot of occupation
plt.show()


# In[20]:


sns.countplot(df.customer_nw_category)
plt.show()


# From above plot,we can say that the customers of medium revenue are high followed by hgh revenue and least is less revenue.

# In[21]:


num_cols = ['customer_nw_category', 'current_balance',
            'previous_month_end_balance', 'average_monthly_balance_prevQ2', 'average_monthly_balance_prevQ',
            'current_month_credit','previous_month_credit', 'current_month_debit', 
            'previous_month_debit','current_month_balance', 'previous_month_balance']


# In[22]:


df[num_cols].hist(bins=15, figsize=(20, 7), layout=(3,4))
plt.show()#histogram plots of numerical columns


# In[23]:


d=df.corr()
plt.figure(figsize=(15,7))
sns.heatmap(d,vmin=-1, vmax=1,cmap="YlGnBu",annot=True)  
plt.show()


# From this heatmap,we can say that the numerical columns are highly correlated.

# In[24]:


df.isnull().sum()#missing values


# We can see that gender,dependents,occupation,city,days_since_last_transaction has missing values.

# Handling the missing values

# In[25]:


df['gender'].value_counts()#gender


# In[26]:


dict_gender = {'Male': 1, 'Female':0}#making male as 1 and female as 0
df.replace({'gender': dict_gender}, inplace = True)

df['gender'] = df['gender'].fillna(-1)#fillling null values as -1 say unkown


# In[27]:


df['dependents'].value_counts()#dependents


# In[28]:


df['dependents'] = df['dependents'].fillna(0)#filling missing values with 0


# In[29]:


df['occupation'].value_counts()#occupation


# In[30]:


df['occupation'] = df['occupation'].fillna('self_employed')#filling missing value with most repeated one i.e., self employed.


# In[31]:


df['city'].value_counts()#city


# In[32]:


df['city'] = df['city'].fillna(1020)#filling missing values with most repeated one i.e, 1020


# In[33]:


df['days_since_last_transaction'].value_counts()


# In[34]:


df['days_since_last_transaction'] = df['days_since_last_transaction'].fillna(500)#filling with greater than a year i.e., 500 days as an assumption


# DataPreprocessing

# occupation feature is converted using dummies known as one hot encoding.

# In[35]:


df = pd.concat([df,pd.get_dummies(df['occupation'],prefix = str('occupation'),prefix_sep='_')],axis = 1)#getting dummies and adding those columns to dataframe


# In[36]:


df.drop(['occupation'],axis=1,inplace=True)#removing occupation column


# We know that there are many outliers in numerical columns like balances.So,we preprocess them using standard scaler.

# In[37]:


from sklearn.preprocessing import StandardScaler


# In[38]:


std = StandardScaler()
scaled = std.fit_transform(df[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)
scaled.head()


# In[41]:


df = df.drop(columns = num_cols,axis = 1)#dropping earlier numeric columns and adding preprocessed features to dataframe
df = df.merge(scaled,left_index=True,right_index=True,how = "left")


# In[42]:


df.head()#the final dataset


# In[43]:


x=df.drop(['churn'],axis=1)#dropping churn


# In[44]:


y=df.churn


# In[45]:


from sklearn.model_selection import  train_test_split#Splitting train and test dataset
xtrain, xtest, ytrain, ytest = train_test_split(x,y,test_size=0.35, random_state=42)
print('Train set shape:',xtrain.shape,ytrain.shape)
print('Test set shape:',xtest.shape,ytest.shape)


# ModelBuilding

# Logistic Regression

# In[46]:


from sklearn.linear_model import LogisticRegression
model1=LogisticRegression()


# In[48]:


model1.fit(xtrain,ytrain)
y_pred1=model1.predict(xtest)


# In[49]:


from sklearn.metrics import  accuracy_score
A1=accuracy_score(ytest,y_pred1)*100
print('The accuracy of Logistic Regression Model is',A1,'%')


# Adaboost classifier

# In[52]:


from sklearn.ensemble import AdaBoostClassifier
model2=AdaBoostClassifier()
model2.fit(xtrain,ytrain)


# In[53]:


y_pred2=model2.predict(xtest)
A2=accuracy_score(ytest,y_pred2)*100
print('The accuracy of AdaBoost Classifier model is',A2,'%')


# RandomForestClassifier

# In[54]:


from sklearn.ensemble import RandomForestClassifier
model3=RandomForestClassifier(n_estimators=100)
model3.fit(xtrain,ytrain)


# In[55]:


y_pred3=model3.predict(xtest)
A3=accuracy_score(ytest,y_pred3)*100
print('The accuracy of Random Forest Classifier Model is',A3,'%')


# In[61]:


comparision=pd.DataFrame({'Actual':ytest,'Predicted':y_pred3})#comparing predicted and acutal values
comparision.head(10)


# From above,we can observe 8 out of 10 observations are predicted correct which implies 80% of accuracy.So,we chose the Random Forest classifier model.

# Summary

# The main idea of this use case is to predict the churn of the bank which shows the loss of customers.Initially,I checked the relation between the features using data visualization.Through it,few feature scaling steps are performed in order the train the model in a perfect way.I splitted the data and checked the model with 3 different types of models and atlast I selected the high accuracy model i.e., Random Forest classifier model.

# In[62]:


import pickle


# In[63]:


pickle.dump(model3,open('model_final.pkl','wb'))


# In[ ]:




