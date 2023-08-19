# 🌟 Player Rating Prediction 🤖⚽

# Group J
* Suzanne James Stephen James - 882376
* Swetha Ravichandran - 881904

# Feature Selection and Player Rating Prediction

Welcome to the Feature Selection and Player Rating Prediction repository! This collection of code snippets demonstrates how to wield the power of machine learning for feature selection and predicting player ratings. Whether you're a sports enthusiast, a data science enthusiast, or both, this repository has something exciting for you!

## 📁 Code Files

### Code 1: Best_Feature_Selector.py

`Best_Feature_Selector.py` is your secret weapon for feature selection using a variety of methods. With options like Pearson correlation, Chi-Squared, RFE, Embedded Logistic Regression (Lasso), Embedded Random Forest, and Embedded Light Gradient Boosting (LightGBM), this module equips you to find the best features for your data. It also offers data preprocessing functions to whip your data into shape.

### Code 2: Model_Training.ipynb

In `Model_Training.ipynb`, you'll unlock the secrets of model training and hyperparameter tuning. Using the functions from `Best_Feature_Selector.py`, this Jupyter Notebook guides you through the process of preprocessing data, selecting features, training a Ridge regression model, and saving your masterpiece with the joblib library.

### Code 3: Streamlit_App.py

Behold, `Streamlit_App.py`! This enchanting script conjures a Streamlit app for interactive player rating predictions. Imagine sliders as your magic wands, allowing you to input player characteristics. Watch as the pre-trained Ridge model brings those characteristics to life with estimated player ratings. It's a spellbinding experience you won't want to miss!

## 🚀 Getting Started

1. Clone this repository to your local machine.

2. Ensure you're armed with the right potions by running:

   ```bash
   pip install numpy pandas scikit-learn lightgbm streamlit joblib

3. Unleash the magic by running the Streamlit app! Navigate to the repository folder in your terminal and utter the incantation:

   ``` bash
    streamlit run Streamlit_App.py
   
Your browser will unveil the app, inviting you to predict player ratings with a wave of your hand—or rather, a slide of your sliders.

If you're eager to learn the art of model training, consult the Model_Training.ipynb notebook. This mystical guide will reveal the secrets of training, tuning, and saving your model using the **Best_Feature_Selector.py** module.

## 📝 Note
Keep the **Best_Feature_Selector.py** module handy—ensure it's in the same directory as the other scripts or accessible in your Python environment.

Remember, these code snippets are designed to educate and inspire. Feel free to customize and experiment to make them your own!

Feel the excitement? Dive into the code and embark on your journey to master the art of feature selection and player rating prediction. May your data always be clean, your features always be relevant, and your models always be accurate! 🧙‍♂️🔮📊🏆🌟
