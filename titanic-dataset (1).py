#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#read the csv file in pandas environment
data=pd.read_csv("titanic.csv")


# In[5]:


#print the data to see any discrepancies
print (data)


# In[6]:


#checking top 10 rows of the dataset
print (data.head(10))


# In[7]:


#printing the statistical details of the titanic dataframe
print (data.describe())


# In[8]:


#finding people who survived and did not survive
print(data['Survived'].value_counts())
#the data shows more people died than who survived!


# In[22]:


#finding average age of people who survived and people who did not survive
print(data.groupby('Survived')['Age'].mean())
#data shows older people did not survive who were made to leave the ship at the last,it shows us younger people left first!


# In[13]:


#finding passenger_class distribution of people who survived and those did not
print(data.groupby('Survived')['Pclass'].value_counts())
#data shows us how higher class people were given priority in leaving the ship leading to them surviving more than the lower class people


# In[37]:


#finding number of male and females who survived and who did not
print(data.groupby('Survived')['Sex'].value_counts())

data[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)

#data shows 74 percent females survived but only 18 percent men survived


# In[17]:


#finding where the passengers embarked on
print(data['Embarked'].value_counts())
#data shows majority embarked from point S


# In[23]:


#finding the mean of the total fare
print(data['Fare'].mean())


# In[39]:


#finding the details of the people who paid maximum fare for the ticket
print(data.query('Fare==Fare.max()'))
#data shows maximum fare for a ticket was 512.3292


# In[ ]:


#finding the details of the people who paid minimum fare for the ticket


# In[40]:


print(data.query('Fare==Fare.min()'))
#data shows minimum fare was 0,it means many people boarded the ship without paying


# In[ ]:




