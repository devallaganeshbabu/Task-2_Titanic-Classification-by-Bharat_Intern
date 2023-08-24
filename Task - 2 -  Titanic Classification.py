#!/usr/bin/env python
# coding: utf-8

# # Bharat Intern - Data Science

# # Task - 2 : Titanic Classification

# ----By Shaik Peeru Soheb

# In[1]:


#Importing Libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Data pre-processing

# In[37]:


#reading in the titanic_train.csv file into a pandas dataframe.
df_train=pd.read_csv('F:/Bharat Intern - Data Science/titanic_train.csv')


# In[38]:


df_train.head()


# In[39]:


df_test1=pd.read_csv('F:/Bharat Intern - Data Science/titanic_test.csv')


# In[40]:


df_test1.head()


# In[41]:


df1=pd.read_csv('F:/Bharat Intern - Data Science/gender_submission.csv')


# In[43]:


df1.head()


# In[44]:


df1.shape


# In[45]:


df_test1.shape


# Survived Variable is missing the test dataset, so we have have other dataframe which contains the survived data for test dataset Here we have to merge both the dataset

# In[46]:


df_test = pd.merge(df1, df_test1, how='left', on='PassengerId')
df_test.head()


# In[47]:


# concatenating the train and test data as a single dataframe

df = pd.concat([df_train, df_test]).reset_index(drop=True)
df.head()


# In[49]:


df.duplicated().sum()


# In[50]:


df.info()


# In[51]:


df.isna().sum()


# # Dealing with the Null Values

# In[52]:


# Variable Cabin has more than 95% of missing data, so it is not useful
# Imputing large number of Null value can lead to biasing, so it is good to drop the column

df.drop(['Cabin'], axis=1, inplace=True)

print('Number of column: ', df.shape[1])


# Imputing the Null values in variable Age

# Segregating the data as groups with PClass(1,2,3), then the mean of the each class will be imputed to the Null in the respective classes. This will help avoiding the biasing of imputed data.

# In[53]:


df.Age.mean()


# In[54]:


df.Pclass.unique()


# In[55]:


# Finding the mean value of each passenger classes using groupby

df.groupby('Pclass')['Age', 'Pclass'].mean()


# In[56]:


def impute_age(cols):
    
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 39
        elif Pclass == 2:
            return 29.50
        else:
            return 25
    else:
        return Age


# In[57]:


# imputing the mean age to null values
df['Age'] = df[['Age', 'Pclass']].apply(impute_age, axis=1)

# Droping the datapoints which have very less Null values (Fare, Embarked)
df.dropna(inplace=True)


# In[58]:


df.isna().sum()


# Dropping the column with unique Identity such as (Name, Ticket, passengerId), which does not help much in Analysing and classifying the Data

# In[59]:


# Dropping the unwanted Columns
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)


# In[60]:


df.head()


# In[61]:


df.shape


# In[62]:


df.dtypes


# In[63]:


df.describe()


# In[64]:


df.describe(include='O').T


# # Visualizing and Exploring the data

# In[65]:


df.Parch.unique()


# In[66]:


sns.set_style("whitegrid")
plt.figure(figsize=(9.5, 24))

plt.subplot(7,2,1)
sns.countplot(data=df, x='Survived')
plt.xticks([0, 1], ['Not-Survived', 'Survived'])
plt.title('Count of Not-Survived[0] and Survived[1]')

plt.subplot(7,2,2)
plt.pie(df['Survived'].value_counts(), labels=['Not-Survived', 'Survived'], explode=[0,0.03], autopct='%1.1f%%', pctdistance=0.80)
# draw circle
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
plt.title('Percentage of Not-Survived[0] and Survived[1]')
plt.legend(loc="upper right")

plt.subplot(7,2,3)
sns.countplot(data=df, x='Pclass')
plt.xticks([0,1,2], ['Pclass-1', 'Pclass-2', 'Pclass-3'])
plt.title('Count of 3 Passenger Classes')

plt.subplot(7,2,4)
plt.pie(df['Pclass'].value_counts(), labels=['Pclass-3', 'Pclass-1', 'Pclass-2'], explode=[0.03, 0, 0], autopct='%1.1f%%', pctdistance=0.80)
# draw circle
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
plt.title('Percentage of 3 Passenger Classes')
plt.legend(loc="upper right")

plt.subplot(7,2,5)
sns.countplot(data=df, x='Sex')
plt.title('Count of Male and Female')

plt.subplot(7,2,6)
plt.pie(df['Sex'].value_counts(), labels=['Male', 'Female'], explode=[0.03, 0], autopct='%1.1f%%', pctdistance=0.80)
# draw circle
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
plt.title('Percentage of Male and Female')
plt.legend(loc="upper right")

plt.subplot(7,2,7)
sns.countplot(data=df, x='SibSp')
plt.title('Count of SibSp')

plt.subplot(7,2,8)
plt.pie(df['SibSp'].value_counts(), labels=[0, 1, 2, 3, 4, 5, 8], autopct='%1.1f%%', pctdistance=0.80)
# draw circle
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
plt.title('Percentage of SibSp')
plt.legend(loc="lower left")

plt.subplot(7,2,9)
sns.countplot(data=df, x='Parch')
plt.title('Count of Parch')

plt.subplot(7,2,10)
plt.pie(df['Parch'].value_counts(), labels=[0, 1, 2, 3, 4, 5, 6, 9], autopct='%1.1f%%', pctdistance=0.80)
# draw circle
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
plt.title('Percentage of Parch')
plt.legend(loc="lower left")

plt.subplot(7,2,11)
sns.countplot(data=df, x='Embarked')
plt.title('Count of Embarked')

plt.subplot(7,2,12)
plt.pie(df['Embarked'].value_counts(), labels=['S','C','Q'], autopct='%1.1f%%', pctdistance=0.80)
# draw circle
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)
plt.title('Percentage of Embarked')
plt.legend(loc="upper right")

plt.tight_layout()
plt.show()


# In[67]:


plt.figure(figsize=(9.5, 4))

plt.subplot(1,2,1)
sns.scatterplot(data=df, x='Age', y='Fare', hue='Pclass', palette='Set1')
plt.title('Scatter')

plt.subplot(1,2,2)
sns.boxplot(data=df, x='Pclass', y='Fare')
plt.title('Count of Male and Female')


plt.tight_layout()
plt.show()


# In[68]:


plt.figure(figsize=(9.5, 3.5))

plt.subplot(1,2,1)
sns.distplot(df['Age'])
plt.title('Distribution of Age')

plt.subplot(1,2,2)
sns.distplot(df['Fare'])
plt.title('Distribution of Fare')

plt.tight_layout()
plt.show()


# Dealing with the Outliers//influencer

# In[69]:


# Filter the highest fare, which we consider as a outlier
df[df['Fare'] > 500]


# In[70]:


# Creating the new data, neglecting the Outlier Fare
df = df[df['Fare'] < 500]

df.reset_index(drop=True, inplace=True)
df.shape


# Visualizing the scatter and Box plot after removing the outliers

# In[71]:


plt.figure(figsize=(9.5, 4))

plt.subplot(1,2,1)
sns.scatterplot(data=df, x='Age', y='Fare', hue='Pclass', palette='Set1')
plt.title('Scatter')

plt.subplot(1,2,2)
sns.boxplot(data=df, x='Pclass', y='Fare')
plt.title('Count of Male and Female')

plt.tight_layout()
plt.show()


# In[72]:


plt.figure(figsize=(9.5,4))

sns.displot(data=df, x="Age", col="Survived", bins=20, multiple="dodge", height=4)
plt.suptitle("Number of (Male & Female) Passengers who Not-survived(0) and survived(1)") 
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75])

plt.tight_layout()
plt.subplots_adjust(wspace=0.3)


# In[73]:


sns.displot(data=df, x="Age", col="Survived", bins=20, hue="Sex", multiple="dodge", height=4)
plt.suptitle("Number of (Male & Female) Passengers who Not-survived(0) and survived(1)") 
plt.xticks([0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75])

plt.tight_layout()


# In[74]:


sns.displot(data=df, x="Age", col="Survived", row="Pclass", aspect=2, height=2.2, bins=20)

plt.suptitle("Distribution of Age by Survived and Pclass")
plt.tight_layout()
plt.subplots_adjust(wspace=0.3)
plt.show()


# In[75]:


df.groupby(["Survived","Sex"])["Sex"].count()


# In[76]:


plt.figure(figsize=(9.5, 4))

plt.subplot(1,2,1)
sns.countplot(data=df, x="Survived", hue="Sex")
plt.xlabel("Survived and Not Survived")
plt.title("Number of (Male & Female)\n who Not-survived(0) and survived(1)")
plt.xticks([0,1], ["Not Survived", "Survived"])

plt.subplot(1,2,2)
plt.pie(df.groupby(["Survived","Sex"])["Sex"].count(), labels=["Female Not-Survived","Male Not-Survived","Female Survived","Male Survived"], autopct='%1.1f%%', pctdistance=0.8)
plt.title("Percentage of (Male & Female)\n who Not-survived(0) and survived(1)")
#plt.legend(loc="lower left")
# draw circle
centre_circle = plt.Circle((0, 0), 0.60, fc='white')
fig = plt.gcf()
# Adding Circle in Pie chart
fig.gca().add_artist(centre_circle)

plt.tight_layout()
plt.subplots_adjust(wspace=0.6)


# In[77]:


df.groupby(["Pclass", "Survived"])["Survived"].count()


# In[79]:


plt.figure(figsize=(9.5, 4))

sns.countplot(data=df, x="Survived", hue="SibSp")
plt.title("Number passenger with SibSb who Not-Survived and survived")
plt.xticks([0,1], ['Not-Survived', 'Survived'])

plt.tight_layout()


# In[80]:


plt.figure(figsize=(9.5, 4))

sns.countplot(data=df, x="Survived", hue="Parch")
plt.title("Number passenger with Parch who Not-Survived and survived")
plt.xticks([0,1], ['Not-Survived', 'Survived'])

plt.tight_layout()
plt.show()


# In[81]:


sns.displot(data=df, x="Sex", row="Embarked", col="Survived", aspect=2, height=2.2)

plt.suptitle("Distribution of Sex by Embarked and Survived")
plt.tight_layout()
plt.subplots_adjust(wspace=0.6)
plt.show()


# In[82]:


g = sns.heatmap(data = df.corr(), annot=True, linewidth=0.5)
g.xaxis.tick_top()
plt.tight_layout()
plt.show()


# Splitting the data as Dependent and Independent variable

# In[83]:


df.shape


# In[84]:


x = df.iloc[:, 1:]
y = df.iloc[:, 0]

print("Shape of Independent Variable: ", x.shape)
print("Shape of Dependent Variable  : ", y.shape)


# # Label Encoding

# Label encoding is a technique used to convert categorical variables into numerical format. Many machine learning algorithms work with numerical inputs. By encoding categorical variables into numerical labels, we enable the algorithms to process and understand the data.

# In[85]:


# Encoding the categorical variable Sex with labels 
x['Sex'] = x['Sex'].replace({'male': 0, 'female': 1})
# Encoding the categorical variable Embarked with labels 
x['Embarked'] = x['Embarked'].replace({'S': 0, 'C': 1, 'Q': 2})


# In[87]:


x.head()


# # Feature Scaling

# Feature scaling is required in many machine learning algorithms to ensure that all features contribute equally to the model's learning process. When features have different scales or units, they can dominate the learning process and influence the model more significantly than others.

# In[88]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)


# In[89]:


x_scaled


# # Splitting the data as training and testing set

# In[90]:


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=22)

print('Shape of training Data  : ', x_train.shape)
print('Shape of Training Target: ', y_train.shape)
print('Shape of Testing Data   : ', x_test.shape)
print('Shape of Testing target : ', y_test.shape)


# # Building the Artificial Neural Network

# In[91]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

ann_model = Sequential()
ann_model.add(Dense(64, input_shape=(7,), activation='relu'))
ann_model.add(Dense(32, activation='relu'))
ann_model.add(Dense(1, activation='sigmoid'))


# In[92]:


ann_model.summary()


# In[93]:


# Define the EarlyStopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

ann_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
H = ann_model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_test, y_test), callbacks=early_stopping)


# In[94]:


loss, accuracy = ann_model.evaluate(x_test, y_test)

print('Test Loss    : ', loss)
print('Test Accuracy: ', accuracy)


# In[96]:


plt.figure(figsize=(9.5, 4))

plt.subplot(1,2,1)
plt.plot(H.history['loss'], label='Loss')
plt.plot(H.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training Validation loss')

plt.subplot(1,2,2)
plt.plot(H.history['accuracy'], label='accuracy')
plt.plot(H.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training Accuracy')

plt.tight_layout()
plt.show()


# # Predicting the Test set

# In[97]:


predictions = ann_model.predict(x_test)
predicted = pd.DataFrame(data=predictions, columns=['Predicted'])

# Convert probabilities to predicted classes
predicted['Predicted_Class'] = predicted['Predicted'].apply(lambda x: 1 if x >= 0.5 else 0)

predicted.head()


# In[101]:


y_test.reset_index(drop=True, inplace=True)

# Compare actual and predicted classifications
result = pd.DataFrame({'Actual': y_test, 'Predicted': predicted['Predicted_Class']})

result.head()


# In[102]:


result['correct'] = result['Actual'] == result['Predicted']
result.head(10)


# In[103]:


result['correct'].value_counts()


# In[104]:


correct_predictions = result['correct'].value_counts()[0]
total_instances = result.shape[0]

percentage = (correct_predictions / total_instances) * 100

print("Percentage of correct predictions: ", percentage)


# The model has correctly predicted 89% percent of classes, as the evaluated score also 89% percent. Hence the model is proved and performed good.
