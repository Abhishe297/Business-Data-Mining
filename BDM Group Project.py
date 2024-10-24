import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale
from sklearn.feature_selection import chi2,f_classif,SelectKBest
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import statsmodels.api as sm
from sklearn.metrics import roc_curve, roc_auc_score

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

df = pd.read_excel('/Users/abhishekpanda/Downloads/Business Analytics/Business Data Mining/default of credit card clients.xls')

# Importing the dataset
df.rename(columns={'default payment next month': 'default'}, inplace=True)
print(df.head())
print(df.columns)
pd.set_option('display.max_columns', None)
print(df.info())
print(df.describe())
print(print(df.isnull().sum()))  # to check for null values

# Pre analysis
# Age distribution
df2 = pd.DataFrame()
df2['age'] = df['AGE']

# Define age bins and corresponding labels
bins = [0, 25, 30, 35, 40, 45, 50, 55, 60, 65, 100]  # Adjust the ranges
labels = ['0-25', '26-30', '31-35','36-40', '41-45', '46-50', '51-55','56-60','61-65', '65+']

# Create the age_group using bins
df2['age_group'] = pd.cut(df['AGE'], bins=bins, labels=labels, right=False)

# Count occurrences
age_group = df2.groupby('age_group').size().reset_index(name='counts')


# Plot
plt.figure(figsize=(10, 6))
sns.barplot(x='age_group', y='counts', data=age_group)
plt.title('Age Group Distribution among members')
plt.xlabel('Age Group')
plt.ylabel('Number of Clients')
plt.xticks(rotation=45)
plt.show()

# On time payments vs defaulters
sns.countplot(x='default', data=df)
plt.title('Distribution of Default Payments')
plt.xticks(ticks=[0, 1], labels=['On Timers', 'Defaulters'])
plt.show()

# Correlation matrix heatmap
plt.figure(figsize=(14, 10))
correlation = df.corr()
sns.heatmap(correlation, cmap='coolwarm', annot=True, fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot
sns.pairplot(df, hue='default', vars=df.columns[1:5])  # Adjust based on features
plt.show()

# Credit Limit of clients
plt.figure(figsize = (14,6))
plt.title('Credit limit of clients')
sns.set_color_codes("pastel")
sns.distplot(df['LIMIT_BAL'], kde=True, bins=150, color="blue")
plt.ylabel('Density in thousands')
plt.xlabel('Card Limit')
plt.show()

# Default amount of credit limit
class_0 = df.loc[df['default'] == 0]["LIMIT_BAL"]
class_1 = df.loc[df['default'] == 1]["LIMIT_BAL"]
plt.figure(figsize = (14,6))
plt.title('Default amount of credit limit  - grouped by Payment Next Month (Density Plot)')
sns.set_color_codes("pastel")
sns.distplot(class_1,kde=True,bins=150, color="red")
sns.distplot(class_0,kde=True,bins=150, color="green")
plt.show()

print('MODELLING')
print('-----------------------------------')
# MODELLING

X = df.drop(['ID', 'default'], axis=1)  # Drop 'ID' and target variable
y = df['default']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Feature importance(Fit a Random Forest model to get feature importance)
print('FEATURE IMPORTANCE')
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
# Get feature importance
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
feature_importances.nlargest(10).plot(kind='barh')
plt.title("Top 10 Feature Importance's")
plt.show()
print('-----------------------------------')


# Create a dictionary to store accuracy results
accuracy_results = {}


print('LOGOSTIC REGRESSION')
# Logistic Regression
model = LogisticRegression()
model.fit(X_train, y_train)
pred = model.predict(X_test)
cm = confusion_matrix(y_test, pred)
print(cm)
sns.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()
print("Classification Report:")
print(classification_report(y_test, pred))
accuracy = accuracy_score(y_test, pred)
accuracy_results['Logistic Regression'] = accuracy
print(f"Model Accuracy: {accuracy:.2f}")
print('-----------------------------------')


# Clustering
# Using K-Means for Clustering
print('K-MEANS CLUSTERING')
# X_cluster = df.drop(['ID', 'default'], axis=1)
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Define the KMeans model
kmeans = KMeans(n_clusters=3, random_state=42)  # You can change the number of clusters if needed
kmeans.fit(X_scaled)
# Add cluster labels to the original data
df['Cluster'] = kmeans.labels_
# Visualize the clusters - using 'LIMIT_BAL' and 'PAY_0' as example features
plt.figure(figsize=(10, 6))
sns.scatterplot(x='LIMIT_BAL', y='PAY_0', hue='Cluster', data=df, palette='Set1')
plt.title("Customer Segmentation Clusters")
plt.xlabel('Credit Limit')
plt.ylabel('Payment Status (Sep)')
plt.legend(title='Cluster')
plt.show()
# Display the number of points in each cluster
cluster_counts = df['Cluster'].value_counts()
print("Cluster Counts:\n", cluster_counts)
# Show the cluster centers (after inverse scaling)
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
print("Cluster Centers:\n", cluster_centers)
# Elbow method to determine the optimal number of clusters
inertia = []
for n in range(1, 11):
    kmeans_test = KMeans(n_clusters=n, random_state=42)
    kmeans_test.fit(X_scaled)
    inertia.append(kmeans_test.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()
print('-----------------------------------')


print('NAIVE BAYES MODEL')
# Naive Bayes model
# X = df.drop(['ID', 'default'], axis=1)  # Drop 'ID' and target variable
# y = df['default']  # Target variable
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and fit the Naive Bayes model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
# Make predictions
y_pred = nb_model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
accuracy_results['Naive Bayes'] = accuracy
print(f"Model Accuracy: {accuracy:.2f}")
print('Classification Report:\n', classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Naive Bayes')
plt.show()
print('-----------------------------------')


# KNN
print('KNN')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and fit the KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)  # You can adjust 'n_neighbors' as needed
knn_model.fit(X_train, y_train)
# Make predictions
y_pred = knn_model.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
accuracy_results['KNN'] = accuracy
print(f"Model Accuracy: {accuracy:.2f}")
print('Classification Report:\n', classification_report(y_test, y_pred))
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - KNN')
plt.show()
print('-----------------------------------')


# Classification and decision tree
# Feature scaling
print('DECISION TREE')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Classification Tree
clf_tree = DecisionTreeClassifier(random_state=42)
clf_tree.fit(X_train, y_train)
# Predictions and evaluation
y_pred_clf = clf_tree.predict(X_test)
accuracy = accuracy_score(y_test, y_pred_clf)
accuracy_results['Decision Tree'] = accuracy
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_clf))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_clf)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Classification Tree')
plt.show()
# Plot the classification tree
plt.figure(figsize=(15, 10))
tree.plot_tree(clf_tree, filled=True, feature_names=df.columns[1:-1], class_names=["No Default", "Default"])
plt.title('Decision Tree - Classification')
plt.show()
print('-----------------------------------')


# Random forest classifier
# Prepare the data
print('RANDOM FOREST CLASSIFIER')
# X = df.drop(['ID', 'default'], axis=1)  # Drop 'ID' and target variable
# y = df['default']  # Target variable
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and fit the Random Forest Classifier
rf_classifier = RandomForestClassifier(random_state=42)
rf_classifier.fit(X_train, y_train)
# Make predictions
y_pred_class = rf_classifier.predict(X_test)
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_class)
accuracy_results['Random Forest'] = accuracy
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_class))
# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_class)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Random Forest Classifier')
plt.show()
print('-----------------------------------')


# Hierarchical Clustering
# Compute the linkage matrix
print('HIERARCHICAL CLUSTRING')
Z = linkage(X_scaled, method='ward')
# Create a dendrogram
plt.figure(figsize=(12, 8))
dendrogram(Z, truncate_mode='level', p=5, leaf_rotation=90)
plt.title('Dendrogram for Hierarchical Clustering')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()
# Fit Agglomerative Clustering
n_clusters = 3  # Choose the number of clusters
hierarchical_clustering = AgglomerativeClustering(n_clusters=n_clusters)
df['Cluster'] = hierarchical_clustering.fit_predict(X_scaled)
# Visualize the clusters
plt.figure(figsize=(12, 8))
sns.scatterplot(x='LIMIT_BAL', y='PAY_0', hue='Cluster', data=df, palette='Set1')
plt.title("Customer Segmentation using Hierarchical Clustering")
plt.xlabel('Credit Limit')
plt.ylabel('Payment Status (PAY_0)')
plt.legend(title='Cluster')
plt.show()
print('-----------------------------------')


# ROC curve Using logit
# Fit the new logistic regression model
print('LOGIT MODEL(LOGISTIC REGRESSION)')
X_new = sm.add_constant(X)
logit_model_new = sm.Logit(y, X).fit()
# summary
print(logit_model_new.summary())
#  ROC curve for the new model
predicted_probabilities_new = logit_model_new.predict(X)
accuracy = accuracy_score(y_test, y_pred)
accuracy_results['Logit'] = accuracy
print(f"Model Accuracy: {accuracy:.2f}")
# Calculate ROC curve and AUC
fpr_new, tpr_new, thresholds_new = roc_curve(y, predicted_probabilities_new)
auc_value_new = roc_auc_score(y, predicted_probabilities_new)
#  ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr_new, tpr_new, color='blue', label=f'New ROC Curve (AUC = {auc_value_new:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
print('-----------------------------------')


# Visualize Model Accuracies
print('MODEL ACCURACY')
plt.figure(figsize=(10, 5))
sns.barplot(x=list(accuracy_results.keys()), y=list(accuracy_results.values()))
plt.title('Comparison of Model Accuracies')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.xticks(rotation=45)
plt.show()

best_model = max(accuracy_results, key=accuracy_results.get)
best_accuracy = accuracy_results[best_model]
print(f"\nBest Model: {best_model} with an accuracy of {best_accuracy:.2f}")

# Sort models by accuracy in descending order
sorted_accuracies = sorted(accuracy_results.items(), key=lambda x: x[1], reverse=True)
print("\nTop 3 Models by Accuracy:")
for i, (model_name, acc) in enumerate(sorted_accuracies[:3], start=1):
    print(f"{i}. {model_name}: {acc:.2f}")
print('-----------------------------------')


# Predicted limit balance
print('UPGRADE SUGGESTION BASED ON PREDICTED LIMIT BALANCE')
X = df.drop(['ID', 'LIMIT_BAL'], axis=1)  # Drop 'ID' and target variable
y = df['LIMIT_BAL']  # Target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and fit the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
# Make predictions
pred = rf_model.predict(X_test)
# Identify IDs where the predicted limit balance is greater than the actual limit balance
X_test_original = pd.DataFrame(X_test, columns=X.columns)  # Convert back to DataFrame to match column names
X_test_original['ID'] = df.iloc[X_test_original.index]['ID'].values  # Get the IDs from the original dataset
X_test_original['Actual_LIMIT_BAL'] = y_test.values  # Add actual limit balance
X_test_original['Predicted_LIMIT_BAL'] = pred  # Add predicted limit balance
# Filter the rows where the predicted limit balance is greater than the actual limit balance
higher_predicted = X_test_original[X_test_original['Predicted_LIMIT_BAL'] > X_test_original['Actual_LIMIT_BAL']]
# Show the IDs of those clients
print("IDs of people whose predicted limit balance is greater than the actual limit balance:")
print(higher_predicted['ID'].values)
print(higher_predicted.shape)
print('-----------------------------------')


print('UPGRADE SUGGESTION BASED ON PREDICTED LIMIT BALANCE ONLY FOR CLIENTS WHO HAVE NOT DEFAULTED')
# Define features and target variable
X = df.drop(['ID', 'LIMIT_BAL'], axis=1)  # Drop 'ID' and target variable
y = df['LIMIT_BAL']  # Target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and fit the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
# Make predictions
pred = rf_model.predict(X_test)
# Create a DataFrame for the predictions
print(pred)
pred_df = pd.DataFrame(pred)
print(pred_df.info())
# Create a DataFrame for the original test data
X_test_original = pd.DataFrame(X_test, columns=X.columns)  # Convert back to DataFrame with column names
X_test_original['ID'] = df.loc[y_test.index, 'ID'].values  # Get the IDs from the original dataset
X_test_original['Actual_LIMIT_BAL'] = y_test.values  # Add actual limit balance
X_test_original['Predicted_LIMIT_BAL'] = pred  # Add predicted limit balance
# Filter the rows where the predicted limit balance is greater than the actual limit balance
higher_predicted = X_test_original[X_test_original['Predicted_LIMIT_BAL'] > X_test_original['Actual_LIMIT_BAL']]
# Select only the relevant columns: ID, Actual_LIMIT_BAL, and Predicted_LIMIT_BAL
higher_predicted_filtered = higher_predicted[['ID', 'Actual_LIMIT_BAL', 'Predicted_LIMIT_BAL']]
# Add default status to the filtered DataFrame (if needed)
higher_predicted_filtered = higher_predicted_filtered.merge(df[['ID', 'default']], on='ID', how='left')  # Merge with default column
# Filter for clients who have not defaulted
if 'default' in higher_predicted_filtered.columns:
    non_defaulted_clients = higher_predicted_filtered[higher_predicted_filtered['default'] == 0]  # Assuming '0' indicates no default
    # Show the IDs of those clients
    print("IDs of clients whose predicted limit balance is greater than the actual limit balance and have not defaulted:")
    print(non_defaulted_clients['ID'].values)
    print(non_defaulted_clients.info())
else:
    print("No non-defaulted clients available.")
print('-----------------------------------')


# Default customers prediction
print('DEFAULT CUSTOMERS PREDICTION')
X = df.drop(['ID', 'default'], axis=1)  # Drop 'ID' and target variable
y = df['default']  # Target variable
# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.8, random_state=42)
# Train logistic regression model
logreg_model = DecisionTreeClassifier()
logreg_model.fit(X_train, y_train)
# Get the predicted probabilities for the test set
df['probability_default'] = logreg_model.predict_proba(X_scaled)[:, 1]
# Get IDs of people who have a high probability of default (e.g., > 0.5)
high_default_ids = df[df['probability_default'] > 0.7]['ID']
# Display the result
print("IDs of people who can default easily:")
print('Out of 30,000 credit card users,',len(high_default_ids) ,'users can default easily.')
print(high_default_ids)
print('-----------------------------------')



# Define features and target variable
X = df.drop(['ID', 'LIMIT_BAL'], axis=1)  # Drop 'ID' and target variable
y = df['LIMIT_BAL'] / 10000  # Scale down the target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create a dictionary to store RMSE results
rmse_results = {}

print('LINEAR REGRESSION')
# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
pred = lr_model.predict(X_test)
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
rmse_results['Linear Regression'] = rmse
print(f"Model RMSE: {rmse:.2f}")
print('-----------------------------------')

print('KNN REGRESSOR')
# KNN Regressor
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
rmse_results['KNN Regressor'] = rmse
print(f"Model RMSE: {rmse:.2f}")
print('-----------------------------------')

print('DECISION TREE REGRESSOR')
# Decision Tree Regressor
dt_regressor = DecisionTreeRegressor(random_state=42)
dt_regressor.fit(X_train, y_train)
y_pred_dt = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_dt)
rmse = np.sqrt(mse)
rmse_results['Decision Tree Regressor'] = rmse
print(f"Model RMSE: {rmse:.2f}")
print('-----------------------------------')

print('RANDOM FOREST REGRESSOR')
# Random Forest Regressor
rf_regressor = RandomForestRegressor(random_state=42)
rf_regressor.fit(X_train, y_train)
y_pred_rf = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred_rf)
rmse = np.sqrt(mse)
rmse_results['Random Forest Regressor'] = rmse
print(f"Model RMSE: {rmse:.2f}")
print('-----------------------------------')


# Predicted limit balance
print('UPGRADE SUGGESTION BASED ON PREDICTED LIMIT BALANCE')
X = df.drop(['ID', 'LIMIT_BAL'], axis=1)  # Drop 'ID' and target variable
y = df['LIMIT_BAL']  # Target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and fit the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
# Make predictions
pred = rf_model.predict(X_test)
# Identify IDs where the predicted limit balance is greater than the actual limit balance
X_test_original = pd.DataFrame(X_test, columns=X.columns)  # Convert back to DataFrame to match column names
X_test_original['ID'] = df.iloc[X_test_original.index]['ID'].values  # Get the IDs from the original dataset
X_test_original['Actual_LIMIT_BAL'] = y_test.values  # Add actual limit balance
X_test_original['Predicted_LIMIT_BAL'] = pred  # Add predicted limit balance
# Filter the rows where the predicted limit balance is greater than the actual limit balance
higher_predicted = X_test_original[X_test_original['Predicted_LIMIT_BAL'] > X_test_original['Actual_LIMIT_BAL']]
# Show the IDs of those clients
print("IDs of people whose predicted limit balance is greater than the actual limit balance:")
print(higher_predicted['ID'].values)
print('Total clients: ', len(higher_predicted))
print('-----------------------------------')


print('UPGRADE SUGGESTION BASED ON PREDICTED LIMIT BALANCE ONLY FOR CLIENTS WHO HAVE NOT DEFAULTED')
# Define features and target variable
X = df.drop(['ID', 'LIMIT_BAL'], axis=1)  # Drop 'ID' and target variable
y = df['LIMIT_BAL']  # Target variable
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# Create and fit the Random Forest Regressor model
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
# Make predictions
pred = rf_model.predict(X_test)
# Create a DataFrame for the predictions
print(pred)
pred_df = pd.DataFrame(pred)
# Create a DataFrame for the original test data
X_test_original = pd.DataFrame(X_test, columns=X.columns)  # Convert back to DataFrame with column names
X_test_original['ID'] = df.loc[y_test.index, 'ID'].values  # Get the IDs from the original dataset
X_test_original['Actual_LIMIT_BAL'] = y_test.values  # Add actual limit balance
X_test_original['Predicted_LIMIT_BAL'] = pred  # Add predicted limit balance
# Filter the rows where the predicted limit balance is greater than the actual limit balance
higher_predicted = X_test_original[X_test_original['Predicted_LIMIT_BAL'] > X_test_original['Actual_LIMIT_BAL']]
# Select only the relevant columns: ID, Actual_LIMIT_BAL, and Predicted_LIMIT_BAL
higher_predicted_filtered = higher_predicted[['ID', 'Actual_LIMIT_BAL', 'Predicted_LIMIT_BAL']]
# Add default status to the filtered DataFrame (if needed)
higher_predicted_filtered = higher_predicted_filtered.merge(df[['ID', 'default']], on='ID', how='left')  # Merge with default column
# Filter for clients who have not defaulted
if 'default' in higher_predicted_filtered.columns:
    non_defaulted_clients = higher_predicted_filtered[higher_predicted_filtered['default'] == 0]  # Assuming '0' indicates no default
    # Show the IDs of those clients
    print("IDs of clients whose predicted limit balance is greater than the actual limit balance and have not defaulted:")
    print(non_defaulted_clients['ID'].values)
    print('Total clients: ',len(non_defaulted_clients))
else:
    print("No non-defaulted clients available.")
print('-----------------------------------')

