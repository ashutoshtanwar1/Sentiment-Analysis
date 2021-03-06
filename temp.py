# Natural Language Processing
from flask import Flask, render_template,request

app = Flask(__name__)

# Importing the libraries
import pandas
import numpy as np

# Importing the dataset
dataset = pandas.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)




## Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)


p=0
@app.route('/')
@app.route('/submit',methods=["GET","POST"])
def profile():
    
    if request.method == "POST":
        ashu= request.form.get("rev")
        ashu1= request.form.get("rev1")
        #Test Values
        
        
        review = re.sub('[^a-zA-Z]', ' ', ashu)
        review = review.lower()
        
        review = review.split()
        
        ps = PorterStemmer()
        
        review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
        
        review = ' '.join(review)
        
        corpus.append(review)
        
        # Creating the Bag of Words model
        from sklearn.feature_extraction.text import CountVectorizer
        cv = CountVectorizer(max_features = 1500)
        X = cv.fit_transform(corpus).toarray()
        y = dataset.iloc[:, 1].values
        
        global  p
         # Splitting the dataset into the Training set and Test set
        X_train=X[:1000,:]
        X_test= X[1000+p,:]
        X_test=X_test.reshape(1,-1)
        
        p=p+1
        # Fitting Naive Bayes to the Training set
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        classifier.fit(X_train, y)

        # Predicting the Test set results
        y_pred = classifier.predict(X_test)
        
        if(y_pred==0):
            return render_template("simple1.html",name = ashu1)
        else:
            return render_template("simple2.html",name = ashu1)
    return render_template("simple.html")


if __name__ == "__main__":
    app.run(debug=False)