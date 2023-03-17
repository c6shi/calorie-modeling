<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML">
</script>

# Calorie Hunters :D
This is a project for the Winter 2023 offering of DSC 80 at UC San Diego, creating models on recipe and review data from food.com. Our exploratory data analysis on this dataset can be found [here](https://c6shi.github.io/calorie-analysis/).

Members: Kurumi Kaneko, Candus Shi

# Framing the Problem

In this project, we try to predict the average rating of a recipe using a regression model.
The response variable is average rating, and we chose to predict this because as a recipe author, after posting your recipe, you may want to update the recipe to get higher average ratings, such as by changing the number of steps, e.g. grouping steps together, or to re-upload the recipe and keep it relevant. Using our model, a recipe author could predict how their changes to their recipe may affect its average rating.
For example, if our model predicts that newer recipes have a higher average rating, one may choose to simply re-upload their recipe while keeping the recipe the same. 

We evaluated our model using root mean squared error (RMSE) over other regression modeling scoring methods such as R^2 and mean squared error (MSE) because RMSE tells us the average residual whereas R^2 focuses on the variation of the predictions. Since we are more interested in the actual value of the predicted average rating, we use RMSE. Additionally, RMSE is in the same units as our response variable and does not amplify large errors as MSE does. 
To follow best modeling practices, we use k-fold cross validation with 5 folds over ten iterations (due to the randomness of splitting training and testing data) and take the maximum RMSE to approximately capture the worst possible performance of our model. 

Based on the purpose of our prediction, we assume that your recipe has already been posted. So, at the time of prediction, we would know the nutritional values of a recipe, the number of steps and ingredients in a recipe, and the duration of how long a recipe has been posted on the site.

# Baseline Model

Our baseline model is a linear regression model on polynomially transformed features, namely all the nutritional values, the number of steps and ingredients:

Features:
- `calories`: total calories
- `total fat %` (Percent Daily Value)
- `sugar %`
- `sodium %`
- `protein %`
- `saturated fat %`
- `total carbohydrate %`
- `n_steps`: number of steps
- `n_ingredients`: number of ingredients

All nine features are quantitative. Since none of the features are categorical, and we wanted to see the model's performance on non-encoded quantitative values, we only transformed the features with a degree 2 polynomial.

Baseline model's maximum RMSE from ten iterations of 5-fold validation: 2.071715377525719

This means that on average, our model predicts an average rating of 1.25-1.30 star ratings above or below the actual average rating. This is not "great" because the ratings are out of 5 stars, so for example it could wrongly predict a 5-star recipe to be 3.8 stars or 6.2 stars (which is not possible). We address this issue in our final model.

# Final Model

Our final model is a decision tree regression model with max tree depth 3, but with only certain nutritional values, the number of steps and ingredients and the submitted date.

Features:
- `calories`
  - due to imperfect multicollinearity, we removed the other nutritional value columns because typically an increase in any of those nutritional facts (sugar, fat, sodium, carbohydrates, etc.) correlates with an increase in calories; i.e., if a variable is highly correlated with another variable, it can lead to unwanted bias in our model.
- `n_steps`
  - fewer steps may imply a simpler recipe and so it may be more likely to attract a higher average rating
- `n_ingredients`
  - fewer ingredients may also imply a simpler recipe and so it may be more likely to attract a higher average rating
- `submitted`: the date the recipe was posted on the site
  - since our prediction is motivated by the idea that how long a recipe has been posted may affect the average rating, we want to include this feature

Feature Engineering:
- `calories`: standardized scaling
- `n_steps`: binarized with threshold > 9
- `n_ingredients`: binned
- `submitted`: the number of years a recipe has been posted since March 13, 2023 (a positive rational number)

Hyperparameter Selection:
One of the main hyperparameters of the decision tree regression model is the max depth, which sets how many levels of the tree the model can have when splitting up the data into its leaf nodes. The regressor model additionally takes the mean of the response variable among each leaf node. 
We used GridSearchCV to test different max depth values from 1 to 9 and got the best max depth parameter value of 3. 

Models:
We tested four models and tested three combinations of column transformations:
1) years_posted, number of ingredients binned, calories standardized
2) years_posted, number of steps binarized (>9), number of ingredients binned, calories standardized
3) polynomial transformations with degree 2 (also determined by GridSearchCV) -> linear regression
4) decision tree regression
5) k-neighbors regression

|model                                                                                                       | max RMSE of combination 1| max RMSE of combination 2| max RMSE of combination 3|
|:-----------------------------------------------------------------------------------------------------------|-------------------------:|-------------------------:|-------------------------:|
|linear regression                                                                                           |1.1066225873529487        |1.1068480647496861        |1.1068499899429582        |
|polynomial transformations with degree 2 (determined by GridSearchCV, ranges 1 to 6) -> linear regression   |1.1049600462451854        |1.105605791763102         |1.1056148708298765        |
|decision tree regression with max depth 3 (determined by GridSearchCV, ranges 1 to 9)                       |1.099172960269645         |1.0990192992719736        |1.0998107405874051        |
|k-neighbors regression with number of neighbors 300 (determined by GridSearchCV, ranges 200 to 400, step 5) |1.099172960269645         |1.0990192992719736        |1.0998107405874051        |


# Fairness Analysis

