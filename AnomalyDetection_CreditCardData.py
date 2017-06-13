##Importing Required Packages
import numpy as np
import pydoop.hdfs as hd
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sbn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score, auc, roc_curve, precision_recall_curve, average_precision_score

##Loading Credit Card Dataset
with hd.open("/user/hduser/creditcard.csv") as f:
    CreditCardData =  pd.read_csv(f, header=0)
    
##Reducing the number of records of Original Dataset incase we wish to work on a smaller subset of Dataset
ReducedData = CreditCardData.iloc[:, :]

##Shape of Credit Card Dataset, i.e. number of rows & columns present in Dataset
print("\nShape of Credit Card Dataset (rows, columns): " + str(ReducedData.shape))

##Removing Duplicate Records (if any)
FinalData = ReducedData.drop_duplicates()
print("\nShape of Credit Card Dataset after removing duplicate records (rows, columns): " + str(FinalData.shape))

##Checking for missing values
print("\nThe total number of missing values for each feild are:")
print(FinalData.isnull().sum())

##Displaying Head, i.e. few starting rows of Dataset
print("\nHead of the dataset is:\n")
print( FinalData.head() )

##Description of Dataset (Mean, Standard Deviation, Maximum, Minimum & various other values)
print("\nDescription of dataset is as follows:\n")
print( FinalData.describe() )

##Using PCA (Principal Component Analysis) to find out two main components
pca = PCA(n_components=2).fit(FinalData)
pca_2d = pca.transform(FinalData)

##Plotting Reference Plot of the given Dataset using the above obtained two main components
plt.style.use('classic')
plt.figure('Reference Plot')
plt.title('Reference Plot')                                  
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c='red')
plt.xlim([-85000, 100000])
plt.ylim([0, 5000])
plt.show()

##Using K-Means Clustering Algorithm on the Final Dataset
kmeans = KMeans(n_clusters=4, max_iter=5000)
kmeans.fit(FinalData)

##Illustrating various clusters formed after running K-Means Clustering Algorithm
plt.style.use('classic')
plt.figure('K-Means with 4 Clusters')
plt.title('K-Means with 4 Clusters')
plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=kmeans.labels_)
plt.xlim([-85000, 100000])
plt.ylim([0, 5000])
plt.show()

##Analyzing Class Count, i.e. number of normal and fake transactions present in Dataset
print("\nClass Count:\n\n" + str(FinalData['Class'].value_counts()) + "\n")

plt.style.use('classic')
plt.figure('Class Count')
plt.title('Class Count')
sbn.countplot(FinalData['Class'], color='violet')
plt.show()

##Plotting Histograms of various features V1 - V28 to determine which of them are important
V_Features = FinalData.ix[:,1:29].columns

plt.style.use('ggplot')

for i, cn in enumerate(FinalData[V_Features]):
    fig, ax = plt.subplots(figsize=(15, 5))
    sbn.distplot(FinalData[cn][FinalData.Class == 0],  bins=50, label='Normal Transactions', ax = ax)
    sbn.distplot(FinalData[cn][FinalData.Class == 1],  bins=50, label='Fraud Transactions', ax = ax)
    ax.set_title('Histogram of Feature: ' + str(cn))
    plt.legend(loc='upper left')
    plt.show()
    
##Finding Correlation between various columns of Dataset
fig, ax = plt.subplots(figsize=(20,10))
correlation = FinalData[FinalData.columns].corr(method='pearson')
sbn.heatmap(correlation, square=True, vmax=1, linewidths=1.0, ax=ax)
ax.set_title('Numeric Columns Correlation')
plt.show()

##Sorting various columns of Dataset on the basis of their absolute values of Coorelation with respect to Class in order to drop less important features
corr_dict = correlation['Class'].to_dict()

for key,val in sorted(corr_dict.items(),key=lambda x:-abs(x[1])):
    print('{0} \t : {1}'.format(key,val))
    
##List of features to be dropped
drop_columns = ['V25','V15', 'V13', 'V26', 'V22', 'Amount', 'V23', 'V24', 'V28', 'Time', 'V20', 'V27', 'V21']
    
##Splitting the Dataset into train data and test
FinalData.insert(len(FinalData.columns),'Cluster', kmeans.labels_)

TrainData, TestData = train_test_split(FinalData, test_size=0.25)

TrainData = TrainData.drop(drop_columns, 1)

TestData = TestData.drop(drop_columns, 1)
TestData_matrix = TestData.as_matrix()

Cluster1 = TrainData[TrainData['Cluster'] == 0].as_matrix()
Cluster2 = TrainData[TrainData['Cluster'] == 1].as_matrix()
Cluster3 = TrainData[TrainData['Cluster'] == 2].as_matrix()
Cluster4 = TrainData[TrainData['Cluster'] == 3].as_matrix()

columns = len(TrainData.columns)
TestData_columns = range(0, columns)
TestData_columns.remove(columns-2)

x_Cluster1 = Cluster1[ : , range(0, columns-2)]
y_Cluster1 = Cluster1[ : , columns-2]

x_Cluster2 = Cluster2[ : , range(0, columns-2)]
y_Cluster2 = Cluster2[ : , columns-2]

x_Cluster3 = Cluster3[ : , range(0, columns-2)]
y_Cluster3 = Cluster3[ : , columns-2]

x_Cluster4 = Cluster4[ : , range(0, columns-2)]
y_Cluster4 = Cluster4[ : , columns-2]

test_x_Cluster = TestData_matrix[ : , TestData_columns]
test_x = TestData_matrix[ : , range(0, columns-2)]
test_y = TestData_matrix[ : , columns-2]

##Using Random Forest Classifier to create our classification model
reg_model_Cluster1 = RandomForestClassifier(criterion='entropy', max_features="auto", n_estimators=200, n_jobs=-1, oob_score=True).fit(x_Cluster1, y_Cluster1)
reg_model_Cluster2 = RandomForestClassifier(criterion='entropy', max_features="auto", n_estimators=200, n_jobs=-1, oob_score=True).fit(x_Cluster2, y_Cluster2)
reg_model_Cluster3 = RandomForestClassifier(criterion='entropy', max_features="auto", n_estimators=200, n_jobs=-1, oob_score=True).fit(x_Cluster3, y_Cluster3)
reg_model_Cluster4 = RandomForestClassifier(criterion='entropy', max_features="auto", n_estimators=200, n_jobs=-1, oob_score=True).fit(x_Cluster4, y_Cluster4)

##Predicting the class of test data using classification model
ctr = 0
predicted = []

for i in test_x_Cluster:
    if int(i[-1]) == 0 :
        predicted = np.append(predicted, reg_model_Cluster1.predict(test_x[ctr].reshape(1, -1))) 
    elif int(i[-1]) == 1 :
        predicted = np.append(predicted, reg_model_Cluster2.predict(test_x[ctr].reshape(1, -1))) 
    elif int(i[-1]) == 2 :
        predicted = np.append(predicted, reg_model_Cluster3.predict(test_x[ctr].reshape(1, -1))) 
    elif int(i[-1]) == 3 :
        predicted = np.append(predicted, reg_model_Cluster4.predict(test_x[ctr].reshape(1, -1))) 
    ctr = ctr + 1

##Analysing Classification model using Classification report and Confusion Matrix
print('\nClassification Report:\n')
target_names = ['Class 0', 'Class 1']
print(metrics.classification_report(test_y, predicted, target_names=target_names))

print('\nConfusion Matrix:\n')
print(metrics.confusion_matrix(test_y, predicted))

##Determining Accuracy of the Classification model
'''
Receiver Operating Characteristics (ROC) Curve:
An ROC curve is a commonly used way to visualize the performance of a binary classifier, meaning a classifier with two 
possible output classes.

Area Under the Curve (AUC):
AUC is (arguably) the best way to summarize its performance in a single number

True Positive Rate: When the actual classification is positive, how often does the classfier predict positive?
True Positive Rate = True Positives/All Positives

False Positive Rate: When the actual classification is negative, how often does the classifier incorrectly predict positive?
False Positive Rate = False Positives/All Negatives
'''
acc_score = accuracy_score(test_y, predicted)
print('\nAccuracy = %0.4f' %acc_score)

false_positive_rate, true_positive_rate, thresholds = roc_curve(test_y, predicted)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('\nArea Under the Receiver Operating Characteristic Curve (AUC) = %0.4f'% roc_auc)


plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.2f'% roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()


precision, recall, thresholds = precision_recall_curve(test_y, predicted)
average_precision = average_precision_score(test_y, predicted)

plt.plot(recall, precision, label='Area Under the Precision-Recall Curve (AUPRC) = %0.4f' % average_precision)
plt.plot([0,1],[1,0],'b--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.25])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision Recall Curve')
plt.legend(loc="lower right")
plt.show()
