# Reddit Flair Detector

A Reddit Flair Detector web application to detect flairs of Reddit India posts using Machine Learning algorithms. The application can be found live at [Reddit Flair Detector](https://redditflaredetect.herokuapp.com/).

### Directory Structure

The directory is a ***Flask*** web application set-up for hosting on *Heroku* servers. The description of files and folders can be found below:

  1. App_Final.py - The Flask Web Application File.
  2. requirements.txt - Containing all Python dependencies of the project.
  3. nltk.txt - Containing all NLTK library needed dependencies.
  4. Procfile - Needed for Heroku Deployment.
  5. Reddit_India_Updated5.csv - Data Crawled from Reddit India after cleaning.
  6. SMOTE_XGB_MODEL_Cleaned_Final2.sav - The Final Model saved after training on Dataset.
  7. Notebooks - This has the three Jupyter Notebook Files for Data Collection(Reddit_Crawl_Script), Data Analysis(Exploratory_Data_Analysis_Final) and Flare Prediction(Final_Reddit_Flare_Detector). Notebooks can be opened in Colaboratory.
  8. Templates - This folder Contains the HTML files used for Building the Web Application's Front End.
  9. Static - Contains the CSS file for Web Application's Front End.
  
### Additional Requirements
Apart from the Dependencies in [requirements](https://github.com/omkaranustoop/Reddit-Flare-Detector/blob/master/requirements.txt),
the following Requirements might be needed depending on the Approach:-

tensorflow==2.2.0rc3, tensorboard==2.2.1

### Codebase

The entire code has been developed using Python programming language, utilizing it's powerful text processing and machine learning modules. The application has been developed using Flask web framework and hosted on Heroku web server.

### Project Execution

  1. Open Terminal
  2. Clone the repository by entering 'git clone https://github.com/omkaranustoop/Reddit-Flare-Detector.git'.
  3. Open IPython Console and set the directory to the cloned repository directory.
  4. Create a `virtualenv` by executing the following command: `virtualenv -p python3 env`.
  5. Activate the `env` virtual environment by executing the follwing command: `source env/bin/activate`.
  6. Enter the cloned repository directory and execute `pip install -r requirements.txt`.
  7. Enter `python` shell and `import nltk`. Execute `nltk.download('stopwords')` and exit the shell.
  8. Now, execute the following command: `python manage.py runserver` and it will point to the `localhost` with the port.
  9. Hit the `IP Address` on a web browser and use the application.
  
  
### Approach

After studying several methods and going through various literatures on text-classification, I decided to Experiment with classical Machine Learning Models like Random Forest and Extreme Gradient Boosted Trees with/without Oversampling and UnderSampling. I also experimented with Deep-Learning Architectures like CNN and LSTM. 

The approach taken for the task is as follows:

  1. Used Reddit Crawl Script to Collect 150 Reddit India SubReddit data for each of the 11 flairs.
  2. The data includes *flair,title,score,id,url,number of comments,body,time of creation,author* and *comments*.
  3. For **comments**, only top level comments are considered in dataset and no sub-comments are present.
  4. The Collected data was saved and loaded in Exploratory Data Analysis file for Cleaning and Analysis.
  5. The ***title, comments*** and ***body*** are cleaned by removing bad symbols and stopwords using `nltk`.
  6. Three types of features are considered for the the given task:
    
    a) Body
    b) Comments
    c) Combining Title, Comments, Url and Body as one feature.

  6. The dataset is split into **75% train** and **25% test** data using `train-test-split` of `scikit-learn`.
  7. Then, the following Pipelines were built to train the dataset:
    
    a) Tf-Idf Vectorizer + SMOTE Oversampling + Random Oversampling
    b) Tf-Idf Vectorizer + SMOTE Oversampling + Xgboost
    c) Tf-Idf Vectorizer + Random Oversampling + Xgboost
    d) Tf-Idf Vectorizer + Random Undersampling + Xgboost
    e) Tf-Idf Vectorizer + Random Forest
    f) Tf-Idf Vectorizer + Xgboost
    g) Tf-Idf Vectorizer + MLP
    
  8. Two Deep-Learning Models were also built after applying Tokenization and Embedding on DataSet:-
    
    a) CNN for Text Classification
    b) LSTM for Text Classification
  9. Training and Testing on the dataset showed the **Xgboost** showed the best testing accuracy of **80.85%** when trained on the combination of **Title + Comments + Body+ Url** feature.
   10. The best model is saved and is used for prediction of the flair from the URL of the post.
    
### Results

#### Title as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.6011904762      |
| Linear SVM                 | 0.6220238095      |
| Logistic Regression        | **0.6339285714**  |
| Random Forest              | 0.6160714286      |
| MLP                        | 0.4970238095      |

#### Body as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.2083333333      |
| Linear SVM                 | 0.2470238095      |
| Logistic Regression        | 0.2619047619      |
| Random Forest              | **0.2767857143**  |
| MLP                        | 0.2113095238      |

#### URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.3005952381      |
| Linear SVM                 | **0.3898809524**  |
| Logistic Regression        | 0.3690476190      |
| Random Forest              | 0.3005952381      |
| MLP                        | 0.3214285714      |

#### Comments as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.5357142857      |
| Linear SVM                 | 0.6190476190      |
| Logistic Regression        | **0.6220238095**  |
| Random Forest              | 0.6011904762      |
| MLP                        | 0.4761904762      |

#### Title + Comments + URL as Feature

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Naive Bayes                | 0.6190476190      |
| Linear SVM                 | 0.7529761905      |
| Logistic Regression        | 0.7470238095      |
| Random Forest              | **0.7797619048**  |
| MLP                        | 0.4940476190      |


### Intuition behind Combined Feature

The features independently showed a test accuracy near to **60%** with the `body` feature giving the worst accuracies during the experiments. Hence, it was excluded in the combined feature set.

