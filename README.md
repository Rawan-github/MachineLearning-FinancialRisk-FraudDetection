# Machine-Learning
Machine Learning Project with two datasets ( Loan Dataset for Regression and Credit Card Dataset for Classification )

Problem Statement: We have two datasets (Financial Risk Assessment) and (Credit Card Fraud Detection) and we want to apply machine learning algorithms on these datasets, Choosing the perfect algorithms that will fit perfectly on them.
Financial Risk Assessment: This dataset concerns predicting financial risk values as a function of a few financial features. Regression models are required to predict values that are in a continuous scale and our goal is to determine the most suitable algorithms in estimating these values with the least amount of error.
Credit Card Fraud Detection: This dataset deals with binary classification of credit card transactions as fraudulent or legit. It requires classification models and has the problem associated with class imbalance, where fraudulent transactions are the minority.

	Financial Risk Assessment Dataset	
1- Data Exploration
Dataset Overview:
 Number of samples: N (2000,37 ) rows & columns.
 Missing Data : No missing data.
Duplicate Entries: No Found duplicates.
Visualizations: 
In RiskScore we found 94 outliers , by calculating the Interquartile Range(IQR).
We used ridge, lasso , sgd ,and knn to get the least mse, and visualized the output to be more clear. 

2- Data Preprocessing: 
Feature Scaling: Quantitative features were scaled using MinMaxScaler function.
Encoding Categorical Features: Converting qualitative features like (EmploymentStatus, EducationLevel) into numerical values using LabelEncoder function .
Feature Selection: Removed irrelevant features (ApplicationDate) for cleaner modeling and better evaluation. 
Drop unnecessary columns: dropped ‘ApplicationDate','BaseInterestRate', 'NumberOfDependents', 'UtilityBillsPaymentHistory', and  'JobTenure’

3- Model Selection and Implementation: 
Linear regression : Predict the target variable and evaluate the model's performance using (MSE), (MAE), and (RMSE).
Ridge regression : Regularized linear model tested multiple alpha values to reduce overfitting.
KNN regressor : tested various k values ranging from 1 to 15  to evaluate performance.
NN regressor : Neural Network got the best mse value ( lowest mse ), by 4 hidden layers and max iterations equal to 2000 iterations. 
Svm regressor : Due to the large dataset, SVM takes a long time running and a higher mse values with poly kernel. 
Ensembling models : testing some methods like voting_regressor , RandomForestRegressor, and StackingRegressor with linear, ridge, lasso, sgd and knn model to test the lowest mse. 


Implementation Details:
Data split into training and testing sets (80% training , 20%testing).
Standardized evaluation metrics (MSE, RMSE, R², MAE) were used for comparison
Added 4 layers to the neural network model and used the activation function Logistic and Relu , changed the max iterations from 1000 to 2000 or 3000 iterations.
Rationale: Linear Regression and Ridge Regression are the better models. They both have very similar performance metrics with low MSE, RMSE, MAE, and high R2 scores.KNN Regression performs poorly in all metrics, with higher errors and a negative R2 score.
linear regression is more simple while ridge regression handles complexities better.
In phase 2 Rationale: after some trials, the NN model is the best to get the lowest mse. Svm takes a lot of time with high mse.

4- Model Evaluation:

1-Linear Regression:
	MSE_Train: 3.7263158256775832
MSE_Test: 4.058476108101882

RMSE_Train: 1.9303667593692093
RMSE_Test: 2.014565985045385

R2_Train: 0.937968712725555
R2_test: 0.9346329174290001

MAE_Train: 1.4991649158556601
MAE_Test: 1.5267375873663556

MSE_TEST VALUES: [110.48151, 85.036155, 74.50665444444444, 70.158513125, 66.6930768, 65.2457547222222, 63.71250571428571, 62.476671874999994, 61.50113172839505, 60.78320539999999, 60.24913933884297, 59.54551687499999, 59.06166591715976, 58.7426731632653]

2-Ridge Regression:
MSE_TRAIN_RIDGE & MSE_TEST_RIDGE : 3.7267465238807334 4.056809177904589
R2_SCORE_TRAIN_RIDGE & R2_SCORE_TEST_RIDGE : 0.9379615429725824 0.9346597655268931
MAE_TRAIN_RIDGE & MAE_TEST_RIDGE 1.4998193507228634 1.5275398278699939

3-KNN Regression:
MSE_TEST VALUES: [110.48151, 85.036155, 74.50665444444444, 70.158513125, 66.6930768, 65.2457547222222, 63.71250571428571, 62.476671874999994, 61.50113172839505, 60.78320539999999, 60.24913933884297, 59.54551687499999, 59.06166591715976, 58.7426731632653]
MSE_Train: 46.4598726171875
MSE_Test: 62.476671874999994
MAE_Train: 5.365245312499999
MAE_Test: 6.239862499999999

R2_Train: 0.22659113186498958
R2_Test: -0.006268771931839279


4-svm regression:
With kernel='poly'
The MSE: 0.0027687290530047683

With kernel='rbf'
The MSE: 0.0016535189833153207

With kernel='linear
MSE value: 0.0015311276381386487

	Credit Card Fraud Detection Dataset

Data Exploration: The dataset contains numerical features derived from credit card transactions. Initial analysis revealed a heavy imbalance in the Class target variable, with fraudulent cases being significantly underrepresented.
Visualization: Approximately 0.17% of transactions are fraudulent, emphasizing the need of imbalance handling.

Data Preprocessing: 

Class Imbalance Handling:
Random Under-Sampling: The data was unbalanced due to then bias in (Class) column that had more zeros than one’s , so we had to use  ( Over Sampling Method ) that uses Random Over Sampler function that makes the minority class instances (1s) by randomly duplicating them until they are equal to the majority class (0s).

Data Cleaning:
Removed duplicate and missing values to ensure data quality.

Feature Scaling:
Applied StandardScaler to standardize features for improved model performance.





Model Selection and Implementation:
 We used LogisticRegression, KNN regressor, SGD, SVM, Neural Network, VotingClassifier, RandomForestClassifier, and StackingClassifier   models because they will work perfectly with this datasets because it depends on Classification

Logistic Regression :
Chosen for its simplicity and interpretability in classification tasks. 
First we imported the sklearn library to import logistic regression function and then we identified the ( X train and test data , Y train and test data ) that the test size will take 20% of the main data, and then we fitted the X and Y train to model so it can learn properly and then we identified the Y predicted using the X test model, and then we got the accuracy of the model using the Y test and Y predicted to see how much is the model accurate


KNN : 
Selected to compare performance against a non-parametric method.
First we imported the sklearn library to import KNeighborsClassifier function and, and then we identified the model and set the K to be 5,  then we identified the ( X train and test data , Y train and test data ) that the test size will take 20% of the main data, and then we fitted the X and Y train to model so it can learn properly and then we identified the Y predicted using the X test model, and then we got the accuracy of the model using the Y test and Y predicted to see how much is the model accurate 



SGD : Selected because it is effective in the case of big and sparse data. Import SGDClassifier from sklearn. Defined X_train, X_test, y_train, and y_test with test size as 20% of the main data. The model was fitted on X_train and y_train. Predict y_pred X_test. Calculate the accuracy using y_test and y_pred.

Neural Network : Selected to work with multi-layer perceptrons for complex patterns.
 importing MLPClassifier from sklearn.Identified X_train, X_test, y_train and y_test with a 20% test size of main data. Model construction by putting hidden layers and specified parameters. For example, activation function to be used, solver criteria.Model fitted with using X_train and y_trainUsed X_test to Predict y_predUsed to calculate the accuracy by using y_test and y_pred. And after trying multiple times the best number of layers the got the highest accuracy was 5 layers (128, 64, 32, 16, 8 ) and 1000 iterations.





SVM linear: Chosen because it can handle linearly separable data.Import SVC from sklearn, set the kernel as 'linear'. Defined X_train, X_test, y_train, and y_test with 20 percent of the main data as a test size.Fit the model with X_train and y_train. Predicted y_pred using X_test.Calculated accuracy using y_test and y_pred.

SVM poly : Chosen for its flexibility in handling nonlinear data. Then import SVC from sklearn and create the kernel as 'poly'. Identified X_train, X_test, y_train, and y_test with test size of 20% of the main data. Fits model to X_train and y_train. Prediction y_pred using X_test. Calculated accuracy using y_test and y_pred.

SVM rbf : Chosen for its power in nonlinear classification tasks.Import SVC from sklearn, set the kernel as 'rbf'. Identified X_train, X_test, y_train, y_test with test size as 20% of the main data. Fit the model on X_train and y_train. Predicted y_pred using X_test. Calculated accuracy using y_test and y_pred.

VotingClassifier : It merges several models to work on performance. Import VotingClassifier from sklearn. initialized the base models. Initializing the voting classifier with incentives of the base models. Here, X_train, X_test, y_train, and y_test are defined where the test size comprises 20% of the main data. The model is fitted on X_train and y_train. Here, the y_pred is predicted using X_test. Accuracy using y_test and y_pred has been computed.

RandomForestClassifier : Chosen because of its ensemble approach to solve the classification problem. Import RandomForestClassifier from sklearn. Defined X_train, X_test, y_train, y_test with a test size of 20% of the main data. Model Parameters: defined the number of trees for the model. Fitted the model using X_train and y_train. Predicted y_pred using X_test. Calculated accuracy using y_test and y_pred.

StackingClassifier : Used to combine the predictions of multiple models. Importing StackingClassifier from sklearn. Creating the base models: Logistic Regression, KNN, and SVM. Built the stacking classifier with base models and defined the meta-model. Identifying X_train, X_test, y_train, and y_test with the test size as 20% of the main data. Fitting the model with X_train and y_train. Predicting the y_pred using X_test. Calculating accuracy with y_test and y_pred.

Rationale:
Logistic Regression: Simple, interpretable, and effective for binary classification with linear decision boundaries.
KNN: Capture of nonlinear relationships; does well with small datasets but requires scaling of the method for sure and hyperparameter tuning to work at their best.
SGD: Efficient in large-scale datasets with sparse features, especially for linear models. Neural Network: Handles complex, nonlinear patterns nicely using multi-layer perceptrons; it requires high computational resources and hyperparameter tuning.
SVM Linear: Handles linearly separable data efficiently, robust to overfitting in high-dimensional spaces. 
SVM Poly: Has the flexibility to capture non-linear relationships through polynomial kernels and can be suitable for moderately complex data.
SVM RBF: A very powerful classifier for nonlinear data, as it is able to map inputs into high-dimensional space.
VotingClassifier: Makes predictions based on multiple models to boost overall performance by reducing overfitting.
RandomForestClassifier: Robust ensemble for high-dimensional data, avoiding overfitting due to bagging and randomness in features.
StackingClassifier: Generalizing predictions from diverse models leverages different model strengths.



Model Evaluation: 


Old report (Phase 1) :
1-Logistic Model Evaluation:
Accuracy: 0.9441624365482234
Precision: 1.0
Recall: 0.8962264150943396
F1-Score: 0.945273631840796
Confusion Matrix:
 [[91  0]
 [11 95]]
Classification Report:
               precision    recall  f1-score   support

           0       0.89      1.00      0.94        91
           1       1.00      0.90      0.95       106

    accuracy                           0.94       197
   macro avg       0.95      0.95      0.94       197
weighted avg       0.95      0.94      0.94       197

New report (Phase 2)  : 

-Logistic Model Evaluation:
Accuracy: 0.9991893701758714
Precision: 0.8928571428571429
Recall: 0.5555555555555556
F1-Score: 0.684931506849315
Confusion Matrix:
 [[56650     6]
 [   40    50]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56656
           1       0.89      0.56      0.68        90

    accuracy                           1.00     56746
   macro avg       0.95      0.78      0.84     56746
weighted avg       1.00      1.00      1.00     56746

Old report : 
2-KNN Model Evaluation:
Accuracy: 0.8984771573604061
Precision: 0.9886363636363636
Recall: 0.8207547169811321
F1-Score: 0.8969072164948454
Confusion Matrix:
 [[90  1]
 [19 87]]
Classification Report:
               precision    recall  f1-score   support

           0       0.83      0.99      0.90        91
           1       0.99      0.82      0.90       106

    accuracy                           0.90       197
   macro avg       0.91      0.90      0.90       197
weighted avg       0.91      0.90      0.90       197
New report : 
-KNN Model Evaluation:
Accuracy: 0.9995946850879357
Precision: 0.971830985915493
Recall: 0.7666666666666667
F1-Score: 0.8571428571428571
Confusion Matrix:
 [[56654     2]
 [   21    69]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56656
           1       0.97      0.77      0.86        90

    accuracy                           1.00     56746
   macro avg       0.99      0.88      0.93     56746
weighted avg       1.00      1.00      1.00     56746

3-SGD Model Evaluation:
Accuracy: 0.9991012582384662
Precision: 0.8823529411764706
Recall: 0.5
F1-Score: 0.6382978723404256
Confusion Matrix:
 [[56650     6]
 [   45    45]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56656
           1       0.88      0.50      0.64        90

    accuracy                           1.00     56746
   macro avg       0.94      0.75      0.82     56746
weighted avg       1.00      1.00      1.00     56746

4-SVM linear Model Evaluation:
Accuracy: 0.9994008388256441
Precision: 0.8589743589743589
Recall: 0.7444444444444445
F1-Score: 0.7976190476190477
Confusion Matrix:
 [[56645    11]
 [   23    67]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56656
           1       0.86      0.74      0.80        90

    accuracy                           1.00     56746
   macro avg       0.93      0.87      0.90     56746
weighted avg       1.00      1.00      1.00     56746


5-SVM RBF Model Evaluation:
Accuracy: 0.9994184612131252
Precision: 0.9830508474576272
Recall: 0.6444444444444445
F1-Score: 0.7785234899328859
Confusion Matrix:
 [[56655     1]
 [   32    58]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56656
           1       0.98      0.64      0.78        90

    accuracy                           1.00     56746
   macro avg       0.99      0.82      0.89     56746
weighted avg       1.00      1.00      1.00     56746







6-SVC Poly Model Evaluation:
Accuracy: 0.9993303492757198
Precision: 0.90625
Recall: 0.6444444444444445
F1-Score: 0.7532467532467533
Confusion Matrix:
 [[56650     6]
 [   32    58]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56656
           1       0.91      0.64      0.75        90

    accuracy                           1.00     56746
   macro avg       0.95      0.82      0.88     56746
weighted avg       1.00      1.00      1.00     56746





7-Neural Network Model Evaluation:
Accuracy: 0.9994184612131252
Precision: 0.8607594936708861
Recall: 0.7555555555555555
F1-Score: 0.8047337278106509
Confusion Matrix:
 [[56645    11]
 [   22    68]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00     56656
           1       0.86      0.76      0.80        90

    accuracy                           1.00     56746
   macro avg       0.93      0.88      0.90     56746
weighted avg       1.00      1.00      1.00     56746


Logistic Accuracy: 0.9992
SGD Classifier Accuracy: 0.9991
KNN Accuracy: 0.9996
NN model Accuracy: 0.9994
SVM Linear Accuracy: 0.9994
SVM Poly Accuracy: 0.9993
SVM RBF Accuracy: 0.9994
Voting Classifier Accuracy: 0.9994
Random Forest Classifier Accuracy: 0.9995
Stacking Classifier Accuracy: 0.9996

Comparison of Models : 

Logistic Regression:
Best for binary classification with linear relationships.
Simple and interpretable.
Poor on nonlinear or complex data.

KNN (K-Nearest Neighbors) :
Captures non-linear relationships.
Sensitive to outliers and scaling.
Slower with large datasets due to computational complexity.

SGD (Stochastic Gradient Descent) :
Scales well with large datasets.
Efficient with sparse and high-dimensional data.
Requires careful tuning of learning rate and convergence criteria.

Neural Network (MLPClassifier) :
Models complex, nonlinear patterns effectively.
Computationally expensive and sensitive to hyperparameters.
Requires a large dataset to avoid overfitting.

SVM (Linear, Poly, RBF) :
Linear: Works perfectly for linearly separable data.
Poly: Handles moderately complex data; high-degree polynomials may cause overfitting.
RBF: Best for nonlinear data; computationally intensive in the case of large datasets.

VotingClassifier:
Improves overall performance by combining several models.
To get effective results, well-performing base models are required.

RandomForestClassifier:
Effortless handling of nonlinear and high-dimensional data.
Not prone to overfitting due to ensemble learning itself.
Sometimes I need tuning for optimal tree depth and number of trees.

StackingClassifier:
It leverages the strengths of multiple models to generalize better. It requires careful selection of base models and a strong meta-model.

Best Model : 
The most suitable general model would be RandomForestClassifier, as it has a good balance in interpretability, robustness, and flexibility with various kinds of data.
SVM RBF or Neural Network can be good options for nonlinear high-dimensional tasks.
For using multiple models with the strengths combined, try StackingClassifier if you are seeking top performance.






Conclusion:

Financial Risk Assessment Dataset: Ridge regression performed slightly better than linear regression which was expected because it utilizes ridge regularization. KNN performed poorly on the KNN regression. This could be attributed to the dataset used being a complex one.
dataset used being a complex one.
Fraud Detection Dataset: Logistic Regression stands out as a good low bias high variance estimator.KNN performed well but needed careful dependency management through tuning hyperparameters.
Limitations:
Financial Risk Assessment: 
The models are based on a linear relationship between features and the target variable.
If the data is not linear, then the performance of the models may not be as it should have been.
Using techniques which are more advanced and with no linear assumptions can improve the results.
Fraud Detection Dataset:
Random Over-sampling used in balancing classes can result in the loss of important information belonging to the majority class, which consists of non-fraud cases.
This can reduce the model's capacity for learning from all the available data effectively.




