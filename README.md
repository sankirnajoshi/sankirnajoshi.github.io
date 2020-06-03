# Data science portfolio by Sankirna Joshi


# TO DO 
Add project links
Add project images below each description

This portfolio is a compilation of notebooks which I created for data analysis or for exploration of machine learning algorithms. A separate category is for separate projects.

* [Topic 1](#1): Stand-alone projects
* [Topic 2](#2): Kaggle competitions
* [Topic 3](#3): Classification problems
* [Topic 4](#4): Regression problems
* [Topic 5](#5): Natural language processing
* [Topic 6](#6): Clustering
* [Topic 7](#7): Deep Learning with Computer Vision

---
<a id='1'></a>
## Stand-alone projects.

<a id='sent'></a>
### Sentiment Analysis Web Application

This is a NLP web application that I built from scratch to perform sentiment analysis. I develop and train a Long-Short-Term-Memory Model (LSTM) to classify user sentiments into one of five classes. Then I deploy the code to a Flask server using the Dash framework. The web application takes in user inputs and performs sentiment analysis in real-time as the text is entered. The code can be found [here](https://github.com/sankirnajoshi/sentiment-app).

<img src="images/sentiment_demo.gif" alt="sent-app" width="800"/>

### Unsupervised learning on the FIFA 20 players

FIFA 20 is a soccer video game and has a rich assortment of players with hundreds of attributes per player. My goal was to develop and understand if the players form any interesting clusters through visualization. I performed dimensionality reduction to 2 dimensions and applied KMeans clustering to see the model form three clusters. The three clusters were interestingly formed by players playing in the three main positions in Soccer: Forwards, Midfielders, Defenders. The report I made for school is available [here](reports/Fifa_20_clustering_analysis.pdf) and the code is available [here](https://www.kaggle.com/damnation/pca-and-clustering-fifa-20-players).

<p float="left">
  <img src="images/fifa_20.png" alt="fifa" width="500"/>
  <img src="images/fifa_20_1.png" alt="fifa_1" width="500"/>
</p>

### Iris exploratory dataset analysis

Performed EDA on the classic IRIS dataset using matplotlib and seaborn. Developed a vanilla classification model and KMeans clustering using sci-kit learn. The analysis is available as a notebook [here](notebooks/Iris_EDA.ipynb) and its rendered [markdown webpage](markdowns/Iris_EDA.md). 

<img src="images/iris.png" alt="iris" width="800"/>

<a id='2'></a>
## Kaggle Competitions.

### Carvana Image Masking Challenge

[Carvana Image Masking Challenge](kaggle.com/c/carvana-image-masking-challenge) was a Kaggle competition sponsored by Carvana - A online car buying platform. We were tasked to remove background from car images and create a masking image for the cars so that Carvana could put the car on to any different background somewhat like Photoshop but without human intervention. It was an interesting challenge and I tried to implement the model [this]() research paper. Its a 100 layer deep neural network model performs well on image segmentation task. Here is a [link]() to my code and [here]() is the report that I wrote as part of Udacity Machine Learning Engineer Nanodegree Capstone project.


---
<a id='3'></a>
## Classification problems.

### Logistic Regression from scratch using numpy

[Notebook]() [webpage]()
Developed a simple logistic regression model in Python from scratch using numpy and trained it by developing Stochastic gradient descent variations like Mini-Batch SGD, RMS Prop, SGD with momentum. Code was developed as part of HSE's Coursera course 1 on the Advanced Machine Learning Specialization. 

### Titanic: Machine Learning from Disaster

[Notebook]() [webpage]()

Titanic: Survival Exploration. In 1912, the ship RMS Titanic struck an iceberg on its maiden voyage and sank, resulting in the deaths of most of its passengers and crew. We will explore a subset of RMS Titanic passenger manifest to determine which features best predict whether someone survived or did not survive. Thus, this is a binary classification problem: based on the available information, we predict whether the passengers survived or not.

### Finding donors for CharityML

[Notebook]() [webpage]()

In Finding donors for CharityML, we take on the data collected from the 1994 U.S. Census. Our goal is to construct a model that accurately predicts whether an individual makes more than $50,000. Thus, making it a supervised classification problem. 

### Bankruptcy Prediction
Developed a bankruptcy prediction model using R. The dataset contains the firm level data from the intersection of COMPUSTAT NORTH AMERICA –annual data and CRSP-daily stock data between 1985 and 2006. We built a logistic regression model

### German Credit Scoring data

The German credit scoring data is a dataset that has extensive information about 1000 individuals from Germany, on the basis of which they have been classified as risky or not. The variable response in the dataset corresponds to the risk label, 1 has been classified as bad and 2 has been classified as good. We explore the model and build a baseline Logistic Regression Model performing variable selection using AIC, and BIC metrics. We then explore tree models such as CART, Bagging, and Random Forests.

---
<a id='4'></a>
## Regression problems.

## Boston Housing Price prediction

[Notebook]() [webpage]()

The Boston housing data was collected in 1978 and each of the 506 entries represent aggregated data about 14 features for homes from various suburbs in Boston, Massachusetts. We develop a regression model to predict housing prices for new houses in a similar locality. Being able to predict the prices of houses could be invaluable information for someone like a real estate agent.

---
<a id='5'></a>
## Natural language processing.

### NLP Sentiment Analysis Web Application
This project is described on the top. Please click [here](#1) to navigate to the project description and code.

### NLP generating names with recurrent neural networks
[Notebook]() [Webpage]()
This notebook shows how to generate new human names from scratch or even modify it to adapt to other objects like say pokemon names. The training dataset contains ~8k human names from different cultures. We can generate new names at random, or even generate names starting with a prefix.


### NLP Parts of Speech [POS] Tagging

[Notebook]() [Webpage]()
In this project, we convert a bunch of word sequences into a bunch of parts-of-speech tags. We develop a Bi-directional LSTM model to look at the sequences and generate POS tags for all the tokens. Tagging parts of speech accurately can significantly improve our language and context understanding and can be a great starting point for a Question Answering machine.

---
<a id='6'></a>
## Clustering

### European Employment Data – Clustering

[Notebook]() [Webpage]()
Clustering is an unsupervised machine learning approach to find distinct clusters or groups within the dataset. In this project, we explore the data about the percentage employment in different industries in European countries during 1979. The purpose of examining this data is to get insight into patterns of employment (if any) amongst European countries in 1970s. We study methods like KMeans, Hierarchical clustering.

---
<a id='7'></a>
## Neural Networks - Computer Vision

### Image Captioning using GANs

[Notebook]() [Webpage]()

One of the most exciting projects I've worked on, Image captioning lies at the intersection of Computer vision and natural language processing. In this project, we develop an image to caption model that can produce a textual description for real world images.


### MNIST digits classification with TensorFlow

[Notebook]() [Webpage]()
We develop a simple logit model, a simple perceptron model, and an MLP model to classify hand-written digits.

### Generating human faces with Adversarial Networks

[Notebook]() [Webpage]()

In this project, we study and develop an Adversarial network to  simulate human faces artificially. We employ a generator to generate new faces and a discriminator to predict if the faces are good enough. The model is trained using the popular Labelled faces in the wild (LFW) dataset.

### Classification of Dogs using Transfer Learning

[Notebook]() [Webpage]()

In this project, we will accept any user-supplied image as input.  If a dog is detected in the image, it will provide an estimate of the dog's breed.  If a human is detected, it will provide an estimate of the dog breed that is most resembling. We train a deep CNN using LFW dataset and the dob subset from the ImageNet competition.
# data-science-portfolio
