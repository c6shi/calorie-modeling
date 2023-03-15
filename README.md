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

We evaluated our model using root mean squared error (RMSE) over other regression modeling scoring methods such as R^2 because ---. 

At the time of prediction, we would know the nutritional values of a recipe, the number of steps and ingredients in a recipe, and the duration of how long a recipe has been posted on the site.

# Baseline Model

Our baseline model is a linear regression model on polynomially transformed features, namely all the nutritional values, the number of steps and ingredients:

Features:
- `calories`
- `total fat %`
- `sugar %`
- `sodium %`
- `protein %`
- `saturated fat %`
- `total carbohydrate %`
- `n_steps`
- `n_ingredients`

All nine features are quantitative. Since none of the features are categorical, and we wanted to see the model's performance on non-encoded quantitative values, we only transformed the features with a degree 2 polynomial.

Baseline model's RMSE on one sample of testing data: 1.25074
Baseline model's average RMSE on 10 iterations of 5-fold cross validated data: 1.30187

This means that on average, our model predicts an average rating of 1.25-1.30 star ratings above or below the actual average rating. This is not "great" because the ratings are out of 5 stars, so for example it could wrongly predict a 5-star recipe to be 3.8 stars or 6.2 stars (which is not possible). We address this issue in our final model.

# Final Model

Our final model is a linear regression model on 
