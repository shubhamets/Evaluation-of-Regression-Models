# Evaluation-of-Regression-Models

# Evaluation-of-Regression-Models
In this project, we conducted an in-depth comparative analysis of multiple regression models to identify the most effective techniques for predicting continuous target variables. Our approach involved systematically applying several standard and advanced regression models, including Linear Regression, Polynomial Regression, desion tree, SVR, XGBoost and Random Forest Regression, to a curated dataset known for its relevance in predicting [specific domain, e.g., real estate prices, sales forecasting, etc.].

Key phases of the project included data preprocessing, model training, and rigorous evaluation using metrics such as Mean Squared Error (MSE), R-squared, and Mean Absolute Error (MAE). Through a combination of cross-validation , we optimized each model's parameters to ensure robust performance.

The project findings revealed that XGBoost performed well. The insights will guide future projects aimed at improving prediction accuracy in similar contexts.

This evaluation not only enhances our understanding of various regression techniques but also assists in selecting the appropriate model based on the specific characteristics of the dataset and prediction requirements.

The data set can be found on "https://raw.githubusercontent.com/amankharwal/Website-data/master/advertising.csv"
The dataset is about the advertising cost incurred by the business on various advertising platforms. Below is the description of all the columns in the dataset:

TV: Advertising cost spent in dollars for advertising on TV;
Radio: Advertising cost spent in dollars for advertising on Radio;
Newspaper: Advertising cost spent in dollars for advertising on Newspaper;
Sales: Number of units sold;

R2 score by XGBregressor : 0.9245619583549144
R2 score by linear_regresion : 0.8718069474344207
R2 score by Polynomial regresion: 0.43744415997187347
R2 score by SVR: 0.8406112402281363
R2 score by Decision tree: 0.8877002133294367
R2 score by Randomforest: 0.8932228393593588
 Average acuracy by K fold cross validation:  0.9232251121094667
Std acuracy by K fold cross validation: 0.06402046861101411
