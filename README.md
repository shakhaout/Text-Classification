# Text-Classification
Assignment: Text Classification
1.	Description of the Assignment
Stack Exchange is a very popular Q&A (Question-and-Answer) based website. We want to analyze some archived data of Stack Exchange using text classification.
The goal of text classification is to identify the topic for a piece of text (news article, web-blog, etc.). Text classification has obvious utility in the age of information overload, and it has become a popular turf for applying machine learning algorithms. In this project, you will have the opportunity to implement k-nearest neighbor and Naive Bayes, apply these to text classification on Stack Exchange sample data, and compare the performances of these techniques.
1.1  k-Nearest Neighbor (k-NN)
1. Implement the k-NN algorithm for text classification. Your goal is to predict the topic for N (initially take small number of rows, e.g., 50 rows from each file and later increase the number during final submission) number of texts/rows/documents from each file in the Test folder. Try the following distance or similarity measures with their corresponding representations:
•	Hamming distance: each document is represented as a boolean vector, where each bit represents whether the corresponding word appears in the document.
•	 Euclidean distance: each document is represented as a numeric vector, where each number represents how many times the corresponding word appears in the document (it could be zero).
•	Cosine similarity with TF-IDF weights (a popular metric in information retrieval): each document is represented by a numeric vector as in the case of euclidean distance. However, now each number is the TF-IDF weight for the corresponding word (as defined below). The similarity between two documents is the dot product of their corresponding vectors, divided by the product of their norms.

2.	Let w be a word, d be a document, and N(d,w) be the number of occurrences of w in d (i.e., the number in the vector as in the case of euclidean distance). TF stands for term frequency, and TF(d,w)=N(d,w)/W(d), where W(d) is the total number of words in d. IDF stands for inverted document frequency, and IDF(d,w)=log(D/C(w)), where D is the total number of documents, and C(w) is the total number of documents that contains the word w; the base for the logarithm is irrelevant, you can use e or 2. The TF-IDF weight for w in d is TF(d,w)*IDF(d,w); this is the number you should put in the vector in Cosine similarity. TF-IDF is a clever heuristic to take into account of the “information content” that each word conveys, so that frequent words like “the” is discounted and document-specific ones are amplified. You can find more details about it online or in standard IR text.

3.	You should try k = 1, k = 3 and k = 5 with each of the representations above. Notice that with a distance measure, the k-nearest neighborhoods are the ones with the smallest distance from the test point, whereas with a similarity measure, they are the ones with the highest similarity scores.

4.	Output the result of running all the three values of k using all the three k-NN techniques into a single file.

1.2 Naive Bayes (NB)
Implement the Naive Bayes algorithm for text classification. Your goal is to predict the topic for N (initially take small number of rows, e.g., 50 rows from each file and later increase the number during final submission) number of texts/rows/documents from each file in the Test folder. Naive Bayes used to be the de facto method for text classification.
1.	Consider all the words of a test text/document/question as independent, then calculate the probability of the statement of being a topic and then pick up the topic which has the highest probability score.

2.	Try different smoothing factors (at least 50 different values).

3.	Output the result of running all the values of smoothing parameter into a single file.

1.3 Comparison and Report Writing
In this part, you will compare between the performance of k-NN classifier and Nave Bayes classifier for text classification. Follow the steps below:
1.	Take the best classifier from k-NN. Chose the best value of k and best measure of distance/similarity that gave the best performance.

2.	Compare the best k-NN with Bayesian classifier. Run 50 times both the K-NN and Bayesian learner. Report average accuracy.


2.	Dataset Description

1.	You can download the zipped file containing data. (link: https://www.dropbox.com/s/1jdct708qk8p6za/Data.zip?dl=0 )

2.	In the training and test folder, there are respective xml files.
3.	The topics.txt contains the name of the topics. For each topic, there should be a training xml file and test xml file in the respective folders.

4.	For both training and test type of files, take every line which starts with “<row” and keep only the “Body” portion of this row. Consider only this portion as a document (or text) and the name of the file as the topic name.
Assignment: Text Classification
1.	Description of the Assignment
Stack Exchange is a very popular Q&A (Question-and-Answer) based website. We want to analyze some archived data of Stack Exchange using text classification.
The goal of text classification is to identify the topic for a piece of text (news article, web-blog, etc.). Text classification has obvious utility in the age of information overload, and it has become a popular turf for applying machine learning algorithms. In this project, you will have the opportunity to implement k-nearest neighbor and Naive Bayes, apply these to text classification on Stack Exchange sample data, and compare the performances of these techniques.
1.1  k-Nearest Neighbor (k-NN)
1. Implement the k-NN algorithm for text classification. Your goal is to predict the topic for N (initially take small number of rows, e.g., 50 rows from each file and later increase the number during final submission) number of texts/rows/documents from each file in the Test folder. Try the following distance or similarity measures with their corresponding representations:
•	Hamming distance: each document is represented as a boolean vector, where each bit represents whether the corresponding word appears in the document.
•	 Euclidean distance: each document is represented as a numeric vector, where each number represents how many times the corresponding word appears in the document (it could be zero).
•	Cosine similarity with TF-IDF weights (a popular metric in information retrieval): each document is represented by a numeric vector as in the case of euclidean distance. However, now each number is the TF-IDF weight for the corresponding word (as defined below). The similarity between two documents is the dot product of their corresponding vectors, divided by the product of their norms.

2.	Let w be a word, d be a document, and N(d,w) be the number of occurrences of w in d (i.e., the number in the vector as in the case of euclidean distance). TF stands for term frequency, and TF(d,w)=N(d,w)/W(d), where W(d) is the total number of words in d. IDF stands for inverted document frequency, and IDF(d,w)=log(D/C(w)), where D is the total number of documents, and C(w) is the total number of documents that contains the word w; the base for the logarithm is irrelevant, you can use e or 2. The TF-IDF weight for w in d is TF(d,w)*IDF(d,w); this is the number you should put in the vector in Cosine similarity. TF-IDF is a clever heuristic to take into account of the “information content” that each word conveys, so that frequent words like “the” is discounted and document-specific ones are amplified. You can find more details about it online or in standard IR text.

3.	You should try k = 1, k = 3 and k = 5 with each of the representations above. Notice that with a distance measure, the k-nearest neighborhoods are the ones with the smallest distance from the test point, whereas with a similarity measure, they are the ones with the highest similarity scores.

4.	Output the result of running all the three values of k using all the three k-NN techniques into a single file.

1.2 Naive Bayes (NB)
Implement the Naive Bayes algorithm for text classification. Your goal is to predict the topic for N (initially take small number of rows, e.g., 50 rows from each file and later increase the number during final submission) number of texts/rows/documents from each file in the Test folder. Naive Bayes used to be the de facto method for text classification.
1.	Consider all the words of a test text/document/question as independent, then calculate the probability of the statement of being a topic and then pick up the topic which has the highest probability score.

2.	Try different smoothing factors (at least 50 different values).

3.	Output the result of running all the values of smoothing parameter into a single file.

1.3 Comparison and Report Writing
In this part, you will compare between the performance of k-NN classifier and Nave Bayes classifier for text classification. Follow the steps below:
1.	Take the best classifier from k-NN. Chose the best value of k and best measure of distance/similarity that gave the best performance.

2.	Compare the best k-NN with Bayesian classifier. Run 50 times both the K-NN and Bayesian learner. Report average accuracy.


2.	Dataset Description

1.	You can download the zipped file containing data. (link: https://www.dropbox.com/s/1jdct708qk8p6za/Data.zip?dl=0 )

2.	In the training and test folder, there are respective xml files.
3.	The topics.txt contains the name of the topics. For each topic, there should be a training xml file and test xml file in the respective folders.

4.	For both training and test type of files, take every line which starts with “<row” and keep only the “Body” portion of this row. Consider only this portion as a document (or text) and the name of the file as the topic name.
