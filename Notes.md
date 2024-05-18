# Machine Learning with Python Course Notes

This file contains the notes I have taken during the course. I decided to add my notes into this repository as an aid for other learners that might be interested. Original notes were done in my personal Notion.

# Supervised vs Unsupervised Learning

**Supervised Learning**

- _Classification_: classifies labelled data
- _Regression_: Predicts trends using previous labelled data
- Has more evaluations methods than unsupervised learning
- Controlled environment

**Unsupervised Learning**

- _Clustering_: Finds patterns and groupings from unlabelled data
- Has fewer evaluation methods than sueprvised learning
- Less controlled Environment

# Intro to Regresison

## What is regression?

- process of predicting a continuous value
- Looks at independent variables (explanatory or causal) and how they impact the final state/goal/metric we try and predict

![Overview](<Lesson Notes Images/Regression/overview.png>)

- The dependent variable - or goal we try and predict has to be continuous, however, the independent variables can be continuous or categorical values

## Types of regression models

- Simple Regression - 1 independent variable used to predict dependent variable. e.g. predict CO2 emission vs Engine size of cars
  - Simple linear regression
  - Simple non-linear regression
- Multiple regression - use of multiple independent variables to predict dependent variable e.g. predict CO2 emission2 vs engine size AND cylinders of cars
  - multiple linear regression
  - multiple non-linear regression

## Applications of regression

- Sales forecasting - predict individuals sales based upon age, education, years of experience
- Satisfaction analysis - Psychology - satisfaction based upon demographic and psychological factors
- House prices - size, bedrooms, etc
- employment income - hours worked, age, sex, field, experience, etc
- Can really be applied across very many domains

## Regression algorithms

- ordinal regression
- poisson regression
- fast forest quantile regression
- linear, polynomial, Lasso, stepwise, ridge regression
- bayesian linear regression
- neural network regression
- decision forest regression
- boosted decision tree regression
- KNN (K-nearest neighbours)

# Simple Linear Regression

- We can look at independent variables and identify one that we want to see if there is a linear relationship to the dependent variable we want to be able to predict
- For instance, we can plot the engine size of cars against the amount of CO2 emissions to see if there is a linear relationship

![Simple Linear Regression Overview](<Lesson Notes Images/Regression/simple_Linear_Regression_Overview.png>)

- In the image above, we can see that as the engine size increases, there is an increase in the amount of CO2 emissions and we can draw a line through our data.
- If the line we draw is accurate and a good measure of our data, we can use it to build a model to predict the amount of CO2 based upon an engine size
- The fit line is traditionally shown as a polynomial
  - $\widehat{Y} = \theta_0 + \theta_1 X_1$
  - $\theta$ = theta
  - $\theta_0$ and $\theta_1$ are the parameters of the line we must adjust. They are the coefficients of the linear equation
  - $\theta_0$ = known as the intercept
  - $\theta_1$ = slope or gradient of the fitting line
  - $\widehat Y$ = response variable/ dependent variable
  - $X_1$ = independent variable

## Simple Linear Regression Equations

2 approaches to adjusting our params to find best fitting line:

- mathematical
- optimisation
- Mathematical - Calculate theta 0 and theta 1 using their prescribed algorithms

### Theta zero and Theta 1

- Mathematical approach to adjusting our params

$$  \large \theta_1 =\space \frac{\large \sum^s_{i=1}(x_i-\bar{x})(y_i - \bar{y})}{ \large \sum^s_{i=1}(x_i-\bar{x})^2} $$

At first, this equation looks psycho but isn’t so bad when walked through slowly and broken down:

<br/>

- $\large \sum^s_{i=1}$ = We want the total sum of the following equation for essentially each row of x and y, indexed as i
  
<br/>

- $\large (x_i-\bar{x})$ = Current value of x, minus the AVERAGE of x —> $\large \bar{x}$ = average of x

<br/>

- $\large (y_i-\bar{y})$ = current value of y (our dependent value) minus the AVERAGE of y —> $\large \bar{y}$ = average of y

<br/>

- Using this information of understanding the equation, we can also make sense of the bottom half of the equation
  
<br/>

$$ \large \theta_0 = \bar{y} -\theta_1 \bar{x} $$

<br/>

- $\large \theta_0$ (theta 0) equation is much easier to understand once you have grasped the $\large \theta_1$ equation - This equation asks us for the AVERAGE of the y values, MINUS the $\large \theta_1$ value from its equation, MULTIPLIED by the AVERAGE value of x values

![Estimate Params](<Lesson Notes Images/Regression/theta_estimate_params.png>)

- We can then make a prediction of our dependent variable based upon the independent variable and use the equations shown above

![Prediction with linear regression](<Lesson Notes Images/Regression/predict_with_params_simple.png>)

## Pros of linear regression

- Very fast
- not tuning of parameters
- easy to understand and highly interpretable

# Model Evaluation in Regression Models

## Model evaluation approaches

- Course will discuss 2 types of evaluation approaches:
  - Train and Test on the same dataset
  - Train/Test Split
- Train and Test on Same Dataset:
  - Complete raining of our model on our data set
  - then we take a portion of that dataset, removing the actual dependent variable value and keeping the independent variable/s
  - With those independent variables, we then pass that into our model to carry out the predictions and compare the result to the ACTUAL values of the dependent variable
  - This will indicate how accurate our model actually is

![Model Evaluation](<Lesson Notes Images/Regression/simple_lin_model_eval.png>)

### Considerations of train and test on the same data set

- Will have High “training accuracy”(on the dataset it was trained on)
- Most likely have Low “out-of-sample-accuracy” (data that was not in the dataset that it was trained on)

## Calculating the accuracy of a model

- There are a number of metrics to use to indicate accuracy, one of the simplest is comparing the values of model and actual values, as discussed above
- We can call this the _Error_ value
- $\large Error/Mean\space Absolute\space Error(MAE)=\frac{1}{n} \displaystyle\sum_{j=1}^n | y_j - \widehat y_j |$
  - $\large y$ = Actual value of dependent variable
  - $\large \widehat y$ = Predicted value of dependent variable
  - All this equation really says is that for each row of data we want to SUBTRACT the PREDICTED y value, FROM the ACTUAL y value and record the SUM
    - Then DIVIDE that sum by the NUMBER OF ROWS OF DATA used —> to give us the average metric of ERROR
  - Essentially, it can be summarised as:
    - calculate the average difference between predicted value and actual value from the sets of data tested

## What is training and out-of-sample accuracy?

- Training accuracy:
  - % of correct predictions when using the training data set
  - Not always a good thing, possible to have result of ‘over-fitting’
    - Over-fit - the model is overtrained to the dataset, which may capture noise and produce a non-generalised model
- Out-of-Sample accuracy
  - % of correct predictions that the model makes on data the model has not been trained on
  - Train and test on same dataset tends to have low accuracy on out-of-sample data due to overfit
  - It’s important that our models have a high, out-of-sample accuracy because our models aim is to make correct predictions on new datasets

## Train/Test split evaluation approach

- This approach sees us split the dataset we have into two sections:
  - A training set
  - Testing set
- This allows us to train our model on the data we have but at the same time reserving some of the data to be tested on as out-of-sample
- This allows us to carry out accuracy tests, like the error metric discussed before

![Train/Test Split](<Lesson Notes Images/Regression/simp_lin_test_split.png>)

![Train/Test Evaluation Approach](<Lesson Notes Images/Regression/simp_lin_test_split_eval.png>)

## K-Fold Cross-Validation and how to use it

- Set aside a percentage of the dataset to use for testing (e.g. first 25%), use the rest for training, then use the next group of percentage for testing and use the rest for training, keep repeating
  - Record the accuracy for each grouped testing, which can be used to calculate the average score

![K-Fold Approach](<Lesson Notes Images/Regression/k-cross-fold-approach.png>)

- Each fold is unique, where no training data used in one fold is used in another
- This method is essentially multiple rounds of Train/Test Split using the same dataset where each split is different

# Evaluation Metrics in Regression Models

- Will review the following metrics of evaluation:
  - MAE - Mean Absolute Error
  - MSE - Mean Squared Error
  - RMSE - Root Mean Squared Error
  - RAE - Relative Absolute Error
  - RSE - Relative Squared Error

## What is an error of the model?

- Error: measure of how far the data is from the fitted regression(trend) line

![What is Error of Model](<Lesson Notes Images/Regression/simpl_lin_error_model.png>)

<br/>

$$\large Error/Mean\space Absolute\space Error(MAE)=\frac{1}{n} \displaystyle\sum_{j=1}^n | y_j - \widehat y_j |$$

<br/>

$$\large Mean\space Squared\space Error(MSE)=\frac{1}{n} \displaystyle\sum_{j=1}^n ( y_j - \widehat y_j )^2 $$

<br/>

- The MSE is often more popular than the MAE because it focusses more on LARGE errors as it exponentially increases large errors in comparison to small ones
- The mean of all residual errors

<br/>

$$ \large Root \space Mean\space Squared\space Error(MSE)=\sqrt {\frac{1}{n} \displaystyle\sum_{j=1}^n ( y_j - \widehat y_j )^2} $$

<br/>

- This is just the squared root of the MSE
- It is one of the most popular metrics because it is interpretable as the same units as the $\large y$ units, making it easy to relate its information

<br/>

$$ \large Relative \space Absolute \space Error(RAE)= \huge \frac{\sum^n_{j=1}|y_j-\widehat y_j|}{\sum^n_{j=1}|y_j-\bar y_j|} $$

<br/>

- $\large \bar y$ = mean value of y (dependent)
- Takes the total absolute error and normalises it by dividing it by the total absolute error of the simple predictor

<br/>

$$\large Relative \space Sqaured\space Error(RSE)= \huge \frac{\sum^n_{j=1}(y_j-\widehat y_j)^2}{\sum^n_{j=1}(y_j-\bar y_j)^2}$$

<br/>

- $\large \bar y$ = mean value of y (dependent)
- Very similar to RAE but is widely adopted by the data science community as it is used for calculating $R^2$

<br/>

$$\large R^2 = 1-RSE $$

<br/>

- $R^2$ is not en error per say, but is a popular metric for the accuracy of your model
- It represents how close the data points are to the fitted regression line
- The higher the value of $R^2$, the better the model fits your data

# Multiple Linear Regression

## Brief Overview

- Multiple Linear regression is having multiple independent variables (x values) impact the dependent variable (y value)
- It is an extension of the simple linear regression
- EXAMPLE: predict CO2 Emissions vs (Engine Size and Num of Cylinders)
- Still have the limitations of dependent variable being a CONTINUOUS value

## Applications of Multiple Linear Regression

- There are two main applications:
  - Identify strength of the effect the independent variables have on the prediction dependent variable
    - EXAMPLE: Does revision time, test anxiety, lecture attendance and gender have any effect on the exam performance of students?
  - Understand how the DEPENDENT variable changes when we change the independent variable
    - EXAMPLE: How much does blood pressure go up or down for every unit increase or decrease in the BMI of a patient?

## Multiple Linear Regression Equations

- Will base the work off the car data we have
- EQUATION FOR MULTIPLE LINEAR REGRESSION
  - $\huge \widehat y=\theta_0+\theta_1x_1+\theta_2x_2+\space ...\theta_nx_n$
    - $\widehat y$ = Predicted y value
    - $\theta_0$ = the intercept
    - $\theta_1$ = theta value for feature set column 1
    - $x_1$ = feature set column 1 value
  - We can also show it as a vector form:
    - $\huge \widehat y=0^TX$
    - $\large \theta ^T$ = Theta transposed

![Multiple Regression Theta Transposed](<Lesson Notes Images/Regression/multiple_regr_theta_transposed.png>)

- X = the feature set. Being the components of the data. E.g X1 = engine size, X2 = num of cylinders, etc.
  - The first element of the feature set is set to 1 because it turns the theta zero into the intercept or bias parameter

![Multiple Regression X values Vector](<Lesson Notes Images/Regression/multiple_regr_x_values.png>)

- When $\large \theta ^TX$ is in a 1 dimensional space it is an equation of a line (like simple linear regression)
  - In higher dimension, where we have more than 1 input, the line is called a plane or a hyper plane and this is what is used for multiple linear regression

## Best fit plane/hyper plane Goal

- The goal is to find the best fitting plane for our data
- To do this we should estimate the values of theta vector that best predict the value of the target field (y) in each row
- To achieve this, we need to minimise the error of the prediction
-

## optimised parameters for theta vector

- Firs thing is to find what the optimised parameters are (e.g engine size, cylinders, number of seats), then we will find a way to optimise the parameters
- OPTIMISED PARAMETERS ARE PARAMETERS THAT LEAD TO THE FEWEST ERRORS

## Using MSE to find the errors in the model - continuation of optimised params

- We will work through using MSE to calculate errors, with the assumption we have already solved $\theta$ (theta) vector values
- We can then use the feature sets (x values) to predict the dependent variable (y value) for a car

![Multiple Regression Demo Data](<Lesson Notes Images/Regression/mult_lin_regr_start_data.png>)

- If we plug the feature sets values into our equation we find $\large \widehat y$ (predicted value)

![Multiple Regression Demo Y Predicted](<Lesson Notes Images/Regression/mult_lin_regr_demo_y_pred.png>)

- We can then compare the ACTUAL value of $\large y_i$

![Multiple Regression Demo Y Actual](<Lesson Notes Images/Regression/mult_lin_regr_demo_y_actual.png>)

- We then find how DIFFERENT the predicted value is from the actual value (**residual error**)
  - $\large y_i -\widehat y_i$

![Multiple Regression Demo Actual vs Predicted](<Lesson Notes Images/Regression/mult_lin_regr_demo_y_actual_vs_pred.png>)

- The mean of all residual errors (for each row of x and y values) can show how bad the model is representing the data set - This is called the Mean Squared Error → which was discussed briefly earlier

<br/>

$$\large Mean\space Squared\space Error(MSE)=\frac{1}{n} \displaystyle\sum_{j=1}^n ( y_j - \widehat y_j )^2 $$

<br/>

## How do we find the parameter or coefficients of multiple linear regression?

### How to estimate $\large \theta$ (theta) vector?

- Few approaches, but the most common methods are:
  - Optimisation algorithm approach
  - Ordinary Least Squares

### Ordinary Least Squares

- Tries to estimate the values of the coefficients by minimising the MSE (mean squared error)
- uses the data as a matrix and uses Linear algebra operations to estimate the optimal values for the theta
- CONSIDERATIONS:
  - It can take a long time to complete this operations over large datasets (10K+ rows) due to the time complexities

### Optimisation algorithm

- Use a process iteratively minimising the error of your model on the training set
- An example of this is **Gradient Decent**
  - starts by using random values for the coefficients and then calculates the errors and tries to minimise it through y’s changing of the coefficients in multiple iterations
  - This is the proper approach if you have a large data set

## Making Predictions with multiple linear regression after finding optimised parameters

![Multiple Regression Making Predictions](<Lesson Notes Images/Regression/mult_lin_regr_pred_post_optim_params.png>)

- The values identified for each $\large \theta_i$ (e.g. theta 0, theta 1, etc, etc) indicate the predicted impact of that variable on the dependent variable
  - So looking at the example above, it shows that the number of cylinders has a higher impact on the prediction compared to the engine size

## Q&A on multiple linear regression

- How to determine whether to use simple or multiple linear regression?
  - sometimes multiple independent variables results in a better model than a simple linear model
- How many independent variables should you use?
  - Adding too many unjustified independent variables can result in an overfit model
- Should independent variables be continuous?
  - categorical independent variables can be incorporated into regression models by converting them into numerical values
    - e.g. TRANSMISSION TYPES
      - manual = 0
      - automatic = 1
- What are the linear relationships between the dependent variable and the independent variables
  - Remember that these approaches require a linear relationship
  - visualising the data with scatter plots can indicate a linear relationship, if there is no linear pattern then it is likely not suitable for linear regression models and should instead use a NON-LINEAR regression model

## Bonus - Explained Variance Regression Score

- Let $\large \hat y$ be the estimated target output, y the corresponding (correct) target output, and Var be the Variance (the square of the standard deviation). Then the explained variance is estimated as follows:
- $\large \texttt{ExplainedVariance}(y, \hat{y}) = 1 - \frac{\large Var\{y-\hat{y}\}}{ \large Var\{y\}}$
- The best possible score is 1.0, the lower values are worse.

---

# Introduction to Classification

## What is Classification

- It is a supervised learning approach
- It consists of categorising some unknown items into a discrete set of categories or "classes"
- Classificatio attempts to examine the relationship of a set of feature variables to a 'target' variable
- The target variable is a categorical variable of discrete values

### How does classification and classifiers work?

Classification determines the class label for an unlabelled test case

![Classification OVerview](<Lesson Notes Images/K-Nearest Neighbours - Classification/classification_overview_1.PNG>)

## Demonstration - Bank wanting to know if loans will be repaid - Binary classifier with 2 classes

We have a set of data of individuals that have previously defaulted on loans that contains feature variables (in this demo it includes things like: age, ed, employ, address, etc) that can be used to know if customers will likely have trouble paying off a loan.

The goal of the model will be to predict whether the individual will default on the loan or not default on the loan. With this being represented as 1 or 0 in the targer variable column.

![Bank Loan Default Model Demo](<Lesson Notes Images/K-Nearest Neighbours - Classification/bank_classification_demo_1.PNG>)

## Multi-Class Classification

Multi-class classification refers to the target variable consiting of multiple options. The example above simply had 1 or 0 as the potential target values, while this approach can handle multiple-classes.

### Multi-Class Classification Demo - Patients of same illness taking different drugs

This demo uses patient data of individuals that are being treated for an illness with varying drugs. The dataset contains features such as: age, sex, blood pressure, etc.

The the target value is the type of drug which is most appropriate for the individual.

We can make a model that can predict what drug should be used for future patients of the same illness.

![Multi-Class Classification Demo](<Lesson Notes Images/K-Nearest Neighbours - Classification/multi-class_demo_drugs_1.PNG>)

## Classification Use Cases

There are many potential business use cases for classification. Some of these are:

- Which category a customer belongs to?
- Whether a customer switches to another brand/provider? (churn)
- Whether a customer responds to a particular advertising campaign?

Data classification has a wide range of applications in a variety of industries. Essentially, many problems can be expressed as association of feature and target variables, especially when labelled data is available.

It can cover areas such as:

- email filtering,
- speech recognition,
- image recognition,
- handwriting recognition,
- bio-metric identification,
- document classification,
- and so much more

![Multi-Class Classification Use Cases](<Lesson Notes Images/K-Nearest Neighbours - Classification/multi-class_use_cases_1.PNG>)

## Classification Algorithms in Machine Learning

This course will only cover a few algorithms, there are many classification algorithms in the world to explore.

- Decision Trees (ID3, C4.5, C5.0)
- Naïve Bayes
- Linear Discriminant Analysis
- K-Nearest Neighbour
- Logistic Regression
- Neural Netwoeks
- Suppoet Vector Machines (SVM)

# K-Nearest Neighbours Algorithm (Classification)

## Intro to KNN (K-Nearest Neighbour)

KNN is an algorithm we can use to predict a classification (dependent variable - Y value) by examing previously labelled independent variables (X values). In other words, it is a method for _classifying_ cases based upon their similarity to other cases.

Cases that are near each other are referred to as _neighbours_.

It is based on the paradigm that similar cases with same classification labels are near each other - thus, measuring the distances away from each other is a measure of their dissimilarity.

There are a number of ways to measure the similarity, or the dissimilarity. This includes using [Euclidian Distance, Manhattan Distance, Minkowski.](https://www.kdnuggets.com/2023/03/distance-metrics-euclidean-manhattan-minkowski-oh.html) (Click on the link to read more about these distances) .

![Eucldian, Manhattan, Minkowski](<Lesson Notes Images/K-Nearest Neighbours - Classification/c_distance_metrics_euclidean_manhattan_minkowski.png>)

### Quick Process Overview

![KNN Overview](<Lesson Notes Images/K-Nearest Neighbours - Classification/knn-intro-1.PNG>)

The process involves looking at the values of the specified independent variables (X values) of the case that we are wanting to predict on, and looking at the N-th (e.g. 1st) closest value in the matching X columns and seeing what value they have in the Y variable (dependent variable) to determine what classification our NEW case should have for the Y value.

![KNN Intro Demo-1](<Lesson Notes Images/K-Nearest Neighbours - Classification/knn-intro-demo-1.PNG>)

## KNN Algorithm

1. Pick a value for K
2. Calculate the sistance of unknown cases from ALL cases
3. Select the K-observations in the training data that are _"nearest"_ to the unknown data point.
4. Predict the response of the unknown data point using the most popular response value from the k-nearest neighbours

### Considerations of the value of K

_K_ is the number of nearest neighbours to reference when examining X (independent values) and the associated classification derived for Y value (dependent variable).

There is consideration to be given to how many neighbours to use in the algorithm because there is such a thing as too few, and too many.

**Too Few**

- If we were to only use the 1st closest neighbour, we are at risk because of the fact that the 1st neighbour may be a very specific case, or potentially an outlier in the data.
- It can result in an overly complex model, which may lead to **_Overfitting_** of the model. Meaning it is not generalised enough to be used for out-of-sample data - or in simpler terms, it can't be trusted to predict unknown samples.

**Too Many**

- If we use too many neighbours, the model becomes overly generalised and increases in potential errors

### Demonstration of K=1 and K=5

In this image, we are using K=1 and we can see that the 1st closest neighbour that matches our X values suggests that we should have our Y value = '4. Total Service'

![KNN Intro Demo-1K](<Lesson Notes Images/K-Nearest Neighbours - Classification/knn-intro-demo-1.PNG>)

However, if we use K=5 we find that the majority of the 5 closest neighbours in X values suggests a different Y value.

Using K=5, we actually have the Y value = '3: Plus Service'. And this makes much more sense then trusting just the first closest.

![KNN Intro Demo-5K](<Lesson Notes Images/K-Nearest Neighbours - Classification/knn-intro-demo-2.PNG>)

## Calculating the similarity/distance in a 1-dimesnional space

We are calculating the distance/similarity between 2 customers with 1 feature (dimension) - being age. Using that data, we can use a few different equations to measure the distance.

![Customer Data](<Lesson Notes Images/K-Nearest Neighbours - Classification/calc-sim-1-dim-data.PNG>)

### Minkowski 1-Dimensional Space

$$\large Dis(x_1,x_2)=\sqrt {\sum^n_{1=0} (x_{1i} - x_{2i})^2}$$

$$\large Dis(x_1,x_2)=\sqrt {(34 - 30)^2} = 4$$

**Breakdown**
$x_1$ and $x_2$ represent the ages of the two customers. In this case, $x_1$ is the age of Customer 1 (which is 34), and $x_2$ is the age of Customer 2 (which is 30).

$i$ is a symbol used to represent each individual component of the ages. Since we're dealing with 1-dimensional data (just one attribute, which is age), $i$ ranges from 0 to 1 in this case. It's just a way of keeping track of which component of the data we're currently looking at.

So, when we see $x_{1i}$ and $x_{2i}$, these represent the respective components (or ages in this case) of Customer 1 and Customer 2 at the $i$th position. Since we only have one attribute (age), there's only one $i$, which takes the value 0.

Think of $i$ as a counter or an index that helps us go through each individual component of the ages. In this specific case, since we're dealing with only one attribute (age), $i$ simply serves the purpose of distinguishing between the two ages.

In more practical terms: When $i=0$, it refers to the first (and only) attribute, which is the age. So, $x_{1,0}$ would represent the age of Customer 1, and $x_{2,0}$ would represent the age of Customer 2.

Since we only have one attribute, $i$ doesn't really change anything in this equation. It's just there as a formality to represent the dimensionality of the data.

In essence, you can think of $i$ as a placeholder that allows us to generalize the equation for cases where we might have multiple attributes to compare, even though in this specific case it doesn't have a practical impact

Putting it all together, the equation calculates the similarity between the ages of Customer 1 and Customer 2. It does this by taking the square root of the sum of the squared differences between each corresponding component of their ages. In simpler terms, it's like measuring how far apart their ages are by subtracting one age from the other, squaring the result (to make sure it's positive), adding up all those squared differences, and then taking the square root of the total. This gives us a single number representing the similarity or dissimilarity of their ages.

## Calculating the similarity/distance in 2-dimensional space

For this example, we still have 2 customers and their ages, however, we also now have their income.

![2-Dimensions Cusotmer Data](<Lesson Notes Images/K-Nearest Neighbours - Classification/calc-sim-2-dim-data.PNG>)

I turns out, we can actually re-use the same Monkowski equation but for a 2 dimensional space.

$$\large Dis(x_1,x_2)=\sqrt {\sum^n_{1=0} (x_{1i} - x_{2i})^2}$$

$$\large Dis(x_1,x_2)=\sqrt {(34 - 30)^2 + (190 - 200)^2} = 10.77$$

**Breakdown**

$x1$ and $x2$ still represent the attributes of the two customers, just like before. However, now we have two attributes: age and income.

$i$ still serves as an index, but now it helps us distinguish between the different attributes (age and income) for each customer.

$x_{1,i}$ and $x_{2,i}$ represent the respective components (age and income) of Customer 1 and Customer 2 at the $i$th position. When $i=0$, it refers to the age, and when $i=1$, it refers to the income.

**_Comparing to the previous 1-dimensional equation_**:

In the 1-dimensional equation, we only had one attribute (age), so $i$ didn't really change anything. It was just there as a formality.

Now, with two attributes (age and income), $i$ becomes more meaningful. When $i=0$, it refers to the age, and when $i=1$, it refers to the income.

So, in this example provided:

- $x_{1,0}$ represents the age of Customer 1.
- $x_{2,0}$ represents the age of Customer 2.
- $x_{1,1}$ represents the income of Customer 1.
- $x_{2,1}$ represents the income of Customer 2.

The equation calculates the distance between the two customers considering both age and income. It squares the difference between each corresponding component, sums them up, and takes the square root of the total to get a single number representing the similarity or dissimilarity between the two customers.

## Calculating the similarity in a multi-dimensional space

After breaking down and understanding the Monkowski equation, we can actually apply the same logic we did from 1 dimension, to 2 dimension, and continue to apply across multiple attribute dimensions!

We have laid the ground work to continue to scale up the number of dimensions.

For this next example, we have 3 dimensions: Age, income, education.

![3-Dimensions Customer Data](<Lesson Notes Images/K-Nearest Neighbours - Classification/calc-sim-3-dim-data.PNG>)

$$\large Dis(x_1,x_2)=\sqrt {\sum^n_{1=0} (x_{1i} - x_{2i})^2}$$

$$\large Dis(x_1,x_2)=\sqrt {(34 - 30)^2 + (190 - 200)^2 + (3 - 8)^2} = 11.87$$

## Finding the Best Value for K

As we discussed earlier, the value of K is incredibly important to avoid overfitting and over generalisation.

So what can we do to find the appropriate value for K?

### Approach to Find K

1. Reserve a portion of your data to measure the accuracy of your model. Like what we did for Linear Regression.
2. Choose K=1 and then use the training portion of the data to calculate the accuracy of prediciton using all samples in the test set
3. Repeat this process, increasing K, and see which K value is best fit for your model

## Computing Continuous Targets Using KNN

KNN can also be used for regression.

In this case, the average or median target value of the nearest neighbours is used to obtain the predicted value for the new case.

### House Price Example

We are looking to predict the price of a home based on its feature set, such as rooms, size, year it was built, etc.

We can very easily find the 3 nearest neighbour houses, not just by distance, but also by the features/attributes we specify and then predict the house price as the median of the neighbours.

# Evaluation Metrics in Classification

With an existing dataset of historical data, the approach to evaluating the accuracy of the model is very similar in principle to what we saw with linear regression.

For each row of data that we predicited the target value of ($\widehat y$), we can compare it to the actual value of $y$.

There are different model evaluation metrics, but this course will talk about 3:

- Jaccard index
- F1-score
- Log loss

## Jaccard Index

$y$ = Actual values/labels
$\widehat y$ = Predicted values/labels

This method views the values of predicted values and actual values like a Venn diagram. Where the intersection of the predicted values circle and actual values circle represent the accuracy of the model.

![Jaccard Venn Diagram](<Lesson Notes Images/K-Nearest Neighbours - Classification/jaccard-venn-overview-1.PNG>)

The equation will produce an output ranging from 0.0, through to 1.0. Where a value of 1.0 is a perfect match and complete intersection of the two circles, while a score of 0.0 shows NO intersection of the two circles - meaning no matches at all.

![Jaccard Venn Diagram vs Index](<Lesson Notes Images/K-Nearest Neighbours - Classification/jaccard-venn-scoring-overview.PNG>)

### Jaccard Index Equation

$$\large j(y,\widehat y) = \huge \frac{|y \cap \widehat y|}{|y \cup \widehat y|} \large  = \huge \frac{|y \cap \widehat y|}{|y|+|\widehat y| - |y \cap \widehat y|}$$

<br/>

![Jaccard Equation Demo](<Lesson Notes Images/K-Nearest Neighbours - Classification/jaccard-equation-demo-1.PNG>)

**Breakdown**

$y$ and $\widehat y$ represent two sets of elements. In many applications, these sets represent the actual labels or categories (ground truth) and the predicted labels or categories, respectively.

$|y|$ represents the number of elements in set $y$, and $|\widehat y|$ represents the number of elements in set $\widehat y$.

$|y \cap \widehat y|$ represents the number of elements that are common to both sets $y$ and $\widehat y$. This is the intersection of the two sets, meaning the elements that appear in both sets.

$|y \cup \widehat y|$ represents the number of elements in the union of sets $y$ and $\widehat y$. This is the total number of unique elements in both sets combined.

So, in simpler terms:

The Jaccard index measures the similarity between two sets by comparing the number of elements they have in common to the total number of unique elements they contain.

The numerator $|y \cap \widehat y|$ counts the common elements, while the denominator $y \cup \widehat y|$ counts all unique elements.

Alternatively, the Jaccard index can be expressed as the size of the intersection divided by the size of the union, as shown in the equation.

The second part of the equation, $|y| + |\widehat y| - |y \cap \widehat y|$, represents the total number of unique elements in both sets, taking into account the elements that are common to both sets only once.

By dividing the size of the intersection by the size of the union, we get a measure of how similar the two sets are. The closer the Jaccard index is to 1, the more similar the sets are, while a value closer to 0 indicates less similarity.

## F1-score (confusion Matrix)

**_The F1-score appraoch can be used for more than 2 labels (binary), however, this will not be covered in this course._**

![F1-score Overview](<Lesson Notes Images/K-Nearest Neighbours - Classification/f1-score-overview.PNG>)

The F1-score evaluation appraoch, represents the $y$ (actual) values as rows, and the $\widehat y$ (predicted) values as columns.

Counting the values across each row will highlight how many counts of that ACTUAL $y$ value there was in the dataset.

The individual counts in the columns are what the model predicted -> $\widehat y$ for that value.

This allows us to see how many correct predicitons were made for each possible $y$ value, and how many were wrong.

For example, looking at the confusion matrix above, we can see that for the churn label 1 (top row) there was a total of 15 counts of this label in the data. The model correctly predicted 6 of these, but it also incorrectly predicted a label of 0 9 times - which is not good. If we look at the bottom row, whcih is chrun label 0, there was a total of 25 counts of this label in our dataset. The model correctly predicted 24 of those, however, it incorrectly predicted a label of 1 for one individual case - which is really good.

Because we have correct predictions against correct values, as well as incorrect predicitons against real values - PLUS we are looking at it with binary of labels, we can convert each square in the matrix into the following:

![F1-score Pos, Neg Labels](<Lesson Notes Images/K-Nearest Neighbours - Classification/f1-score-pos-neg-labels.PNG>)

**True Positive (TP):**

- This happens when we correctly identify something as positive when it actually is positive.
- For example, in medical testing, if someone has a disease and the test correctly identifies them as having the disease, that's a true positive.

**False Negative (FN):**

- This occurs when we wrongly classify something as negative when it's actually positive.
- Using the medical testing example, if someone has a disease but the test says they don't, that's a false negative.

**False Positive (FP):**

- This happens when we incorrectly classify something as positive when it's actually negative.
- In medical testing, if someone doesn't have a disease but the test incorrectly indicates they do, that's a false positive.

**True Negative (TN):**

- This occurs when we correctly identify something as negative when it actually is negative.
- In the medical testing example, if someone doesn't have the disease and the test correctly says they don't, that's a true negative.

The above image has the positive and negative labels set when assessing '_Churn = 1'_

### Precision and Recall

We can use the precision and recall operations for each possible label value (AKA row of the matrix) to help us calculate the F1-score for each label, _based upon the precision and recall of that label._

**Precision**

Precison is a measure of the accuracy, provided that a class label has been predicted.

$ \large Precision = \frac{TP}{(TP + FP)}$ for positive labels

$ \large Precision = \frac{TN}{(TN + FN)}$ for the negative/opposite labels

**Recall**

Recall is the True Positive rate.

$ \large Recall = \frac{TP}{(TP + FN)}$ for positive labels

$ \large Recall = \frac{TN}{(TN + FP)}$ for negative labels

**EXAMPLE FROM IMAGE**

churn 0 -> Precision = 0.73, Recall = 0.96

chrun 1 -> Precisoin = 0.86, Recall = 0.40

### Calculate F1-score from Precisoin and Recall

With the precision and recall values for each label identified, we can calculate the F1-score.

The F1-score is the harmonic average of the precision and recall.

$$\large F1\space score = \frac{2\times (precision \times recall)}{precision + recall}$$

**The F1-score will return a value between 0.0 and 1.0, where the higher the number the higher the accuracy.**

After calculating the F1-score for each class label, or $y$ label. We get the mean of the two scores to find the **_Average Accuracy_**.

**EXAMPLE FROM IMAGE:**

churn 0 -> Precision = 0.73, Recall = 0.96, f1-score = 0.83
chrun 1 -> Precisoin = 0.86, Recall = 0.40, f1-score = 0.55

**_Average Accuracy_** = (0.83 + 0.55) / 2 = 0.72

**Harmonic Average:**

The harmonic average is a way to find the average of numbers when you're dealing with rates or frequencies. It gives more weight to smaller numbers.

In simple words, it's like finding a balance between different rates or frequencies, considering how much each one contributes, especially when some rates are small compared to others.

## Log Loss

Log loss, also known as logarithmic loss or cross-entropy loss, is a measure used to evaluate the performance of a classification model.

Sometimes the output of a classifier is the **_Probability_** of a class label (y value), instead of the label. It assesses how well the probabilities predicted by the model match the actual binary outcomes

For example, customer churn is binary labelled (yes/no or 1/0) where either a custom will leave or not. But we might be interested to know what the probability was for the model to assign the cusotmer the label.

_Log loss measures the performance of a classifier where the predicted output is a probability value between 0 and 1_

**What it assesses:**

- Log loss quantifies the accuracy of the probabilities predicted by the model.
- It penalizes models more heavily for confidently wrong predictions.
- Lower log loss values indicate better performance, with 0 being the best possible score

**In simpler terms:**

- Imagine you're making bets on whether it will rain tomorrow or not, and you assign probabilities to each outcome (like 0.8 for rain and 0.2 for no rain).
- Log loss measures how good you are at estimating these probabilities. If you're confident about rain and it doesn't rain, you'll get a high log loss. If you're confident and it does rain, your log loss will be low.
- So, log loss helps assess how well a model's confidence aligns with reality

This can be helpful in assessing performance of the model prediction because, like in the example below, we may have predicited a label of '1' but the probaility of that case receiving the assinged label is very low - 0.13 in the image below. This indicates a high log loss.

![Log Loss Demo](<Lesson Notes Images/K-Nearest Neighbours - Classification/log-loss-demo.PNG>)

### Calculating the Log Loss

For each row in our dataset, we can use the following equation to calculate the log loss which measures how far away each prediciton is from the actual label:

$$ \large (y \times log(\widehat y)+(1-y)\times log(1-\widehat y))$$

Then, we calculate the average log log across all the rows and losses:

$$ \large LogLoss = - \frac{1}{n} \sum (y \times log(\widehat y)+(1-y)\times log(1-\widehat y)) $$

<br/>

![Logg Loss Equation Demo](<Lesson Notes Images/K-Nearest Neighbours - Classification/log-loss-equation-demo.PNG>)


# Decision Trees

## Introduction

Decision trees allow us to map out all decision paths leading to the classifier we are interested in. The multiple variables can be examined to determine what relational paths exist across variables, leading to the classifier that we will want to predict.



![Decision Tree Intro Demo](<Lesson Notes Images/Decision-Trees/Demo-1.PNG>)

The decision points are referred to as nodes and each **internal node** corresponds to a test - e.g Sex, Cholestrol

Each **branch** - The decision paths stemming from a node - corresponds to a result of the test

Each **Leaf Node** assigns a classifcation - e.g. Drug A, Drug B

![Decision Tree Terminology Demo](<Lesson Notes Images/Decision-Trees/terminology-demo-2.PNG>)

## Decision Tree Learning Algorithm

1. Choose an attribute from your dataset
2. Calculate the significance of attribute in splitting of data (we will go over this soon)
3. Split the data based on the value of the best attribute
4. The, go to each branch and repeat - Going back to step 1

## Building a Decision Tree

Decision trees are built using recurssive partitioning to declassify the data. The most important part is determing the best attribute to use to split the data.

What we are trying to do is find the attribute that BEST gives us insight into the classification that we want to predict is. For example, if we looked an attribute of 'Blood Pressure' and the classification we want to predict on is 'Drug' but when we split our data based upon 'Blood Pressure' and then examine the 'Drug' we see a fairly even distribution of 'Drug' then 'Blood Pressue' is not the best attribute to help us. However, if we split our data based on 'Sex' and we happen to see a significant lean on distribution of 'Drug' than that is a much better attribute.

In the example image below, the Sex attribute value of Female gives us high certainty on the Drug to be assinged, however, its not as strong for Male Sex- but it is still a better attribute to use than something that had a near even spread of drug classification. We can call this **more pure** - there is a majority of a classifcation in the branches. We are looking for the best attributes to **decrease** the impurity in the leaves after splitting them up, based on that feature.

![Best Attribute Demo](<Lesson Notes Images/Decision-Trees/best-attribute-demo-3.PNG>)

We can then repeat this process for each path of the decision and look over the other features available to again help us split our data into the ***purest*** classification groupings we can.

For example, if we use 'Cholesterol' feature, we can fully split our data in the purest form of drug classification of either Drug A or Drug B.

![Repeating Process for Branches](<Lesson Notes Images/Decision-Trees/best-attribute-repeat-Step-in-Branch-demo-4.PNG>)

A ***Pure Node*** is when the splitting of data based upon an attribute results in 100% assignment of target classification feature.
