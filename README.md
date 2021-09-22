# Basic NLP using NLTK and scikit-learn


## What you built? 

A "TextAnalyzer" class that is able to take a csv file of text message information and perform some basic analyses:

* Words commonly used by senders from specific countries (term-frequency analysis)
* "Important" words commonly used by senders from specific countries (term-frequency inverse-document-frequency analysis) 
* Prediction of sender's country of origin based on message (using Multinomial Naive Bayes classifier) 

Below are example outputs for each (which can also be obtained by running the tests.py file): 

* [Term-frequency analysis](./example1.jpg)  
* [TF-IDF analysis](./example2.jpg)
* [MNB classifier prediction](./example3.jpg)

## What you learned

The project as a whole worked, mostly because I closely followed the tutorial/example code. It's a bit un-sophisticated and not too useful (for example, the country of origin prediction classifier works better than guessing but is still not very accurate overall). However, it was a good, brief intro to NLP, particularly in regards to the grunt work needed to clean and pre-process data (given the abundance of NLP libraries out there, this is usually the hardest part).

It was a valuable experience, given that I am thinking of doing something related to AI-powered software testing (which would require NLP to read and understand code).

## Authors

Brandon Feng

## Acknowledgments

CodeAcademy Python NLP Tutorial: 
https://www.codecademy.com/learn/paths/natural-language-processing 

Someone's attempt at completing the final project for the above tutorial: 
https://github.com/macaler/nlp_portfolio_project/blob/main/nlp_portfolio_project_MAC.ipynb 
