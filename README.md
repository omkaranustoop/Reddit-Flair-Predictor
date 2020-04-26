# Reddit Flair Detector

A Reddit Flair Detector web application to detect flairs of Reddit India posts using Machine Learning algorithms. The application can be found live at [Reddit Flair Detector](https://redditflaredetectorfinal.herokuapp.com/).

### Directory Structure

The directory is a ***Flask*** web application set-up for hosting on *Heroku* servers. The description of files and folders can be found below:

  1. **App_Final.py** - The Flask Web Application File.
  2. **requirements.txt** - Containing all Python dependencies of the project.
  3. **nltk.txt** - Containing all NLTK library needed dependencies.
  4. **Procfile** - Needed for Heroku Deployment.
  5. **Reddit_India_Updated5.csv** - Data Crawled from Reddit India after cleaning.
  7. **Pickle_RL_Model_Final.pkl** - Xgboost Model with 82.14 % Accuracy
  8. **Notebooks** - This has the three Jupyter Notebook Files for Data Collection(**Reddit_Crawl_Script**), Data Analysis    (**Exploratory_Data_Analysis_Final**) and Flare Prediction(**Final_Reddit_Flare_Detector**). Notebooks can be opened in Colaboratory.
  9. **Templates** - This folder Contains the HTML files used for Building the Web Application's Front End.
  10. **Static** - Contains the CSS file for Web Application's Front End.
  
### Additional Requirements
Apart from the Dependencies in [requirements](https://github.com/omkaranustoop/Reddit-Flare-Detector/blob/master/requirements.txt),
the following Requirements might be needed depending on whether keras library and tensorflow is used by the user :

tensorflow==2.2.0rc3, tensorboard==2.2.1

### Codebase

The entire code has been developed using Python programming language, utilizing it's powerful text processing and machine learning modules. The application has been developed using Flask web framework and hosted on Heroku web server.

### Project Execution

  1. Open Terminal
  2. Clone the repository by entering 'git clone https://github.com/omkaranustoop/Reddit-Flare-Detector.git'.
  3. Open IPython Console and set the directory to the cloned repository directory.
  4. Execute 'pip install -r requirements.txt' in the console.
  5. Run the App_Final.py file and it will point to the 'localhost' with the port.
  6. Open the 'localhost IP address' on a web browser and use the application.
  
  **Note** - User needs to use own Client id, Client Secret and User Agent in Reddit Crawl Script and App_Final.py
  
  
### Approach

After studying several methods and going through various literatures on text-classification, I decided to Experiment with classical Machine Learning Models like Random Forest and Extreme Gradient Boosted Trees with/without Oversampling and UnderSampling. I also experimented with Deep-Learning Architectures like CNN and LSTM. 

The approach taken for the task is as follows:

**For Collecting Data**(Details in [Script](https://github.com/omkaranustoop/Reddit-Flare-Detector/blob/master/Notebooks/Reddit_Crawl_Script%20.ipynb) )

  1. Used Reddit Crawl Script to Collect 150 Reddit India SubReddit data for each of the 11 flairs.
 
**For Analysing the Collected Data**(Details in [DataAnalysis](https://github.com/omkaranustoop/Reddit-Flare-Detector/blob/master/Notebooks/Exploratory_Data_Analysis_Final%20.ipynb) )

  2. The data includes *flair, title, score, id, url, number of comments, body, time of creation, author* and *comments*.
  3. For **comments**, only top level comments are considered in dataset and no sub-comments are present.
  4. The Collected data was saved and loaded in Exploratory Data Analysis file for Cleaning and Analysis.
  5. The ***title, comments*** and ***body*** are cleaned by removing bad symbols and stopwords using `nltk`.
  
**For Predicting the Flair of Reddit India Posts**(Details in [Detector](https://github.com/omkaranustoop/Reddit-Flare-Detector/blob/master/Notebooks/Final_Reddit_Flair_Detector.ipynb) )

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
    
  9.  Experimentation showed that combination of **Title + Body + Url + Comments** gave best results.
  10. Training and Testing on the dataset showed that **Xgboost** showed the best testing accuracy of **82.14%** when trained on the combination of **Title + Comments + Body+ Url** feature.
  12. The Xgboost model was Saved and used in App Deployment.
    
### Results

#### Machine Learning Models

| Machine Learning Algorithm | Test Accuracy     |
| -------------              |:-----------------:|
| Random Forest + SMOTE      |  0.7256198347     |
| Xgboost + SMOTE            |**0.7933884297**   |
| Xgboost + ROS              |  0.7851239669     |
| Xgboost + RUS              |  0.7917355371     |
| Xgboost                    |**0.8214876033**   |
| Random Forest              |  0.7669421487     |


#### Deep Learning Models

| Model                      | Test Accuracy     |
| -------------              |:-----------------:|
| CNN                        | 0.6553            |
| LSTM                       | 0.1990            |

#### Analysis and Key Observations

