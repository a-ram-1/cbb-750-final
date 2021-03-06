# CBB 750 final project

Last updated 5/6/21

### Classifying cardiomyopathy v. coronary artery disease for patients with and without vectorized text data

A Ram, Jason Liu, Saejeong Park, Sarah Dudgeon

## Here's the basic outline of our project.

* We input subject ids and their associated ICD9 codes, NOTEEVENTS data and processed LABEVENTS ["non-text"] data
* We extract the most recent notes for each patient and run [Doc2Vec](https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html) to create vectors out of each patient's most recent notes
  * We consolidate this into a "text data" matrix
* We combine the text and non-text data matrices by subject id into a new "text+non-text" data matrix
* We generate labels for text, non-text and text+non-text data matrices based on each patient's ICD9 codes
* We classify using GLM [logistic regression], decision tree, SVM, random forest and majority vote
  * We run GLM and decision tree with [AdaBoost](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html) to see if this will improve performance
  * We run majority vote to see if a consensus approach will improve performance. We use all other classifiers for the vote, namely:
    * GLM 
    * GLM with boosting
    * Decision tree
    * Decision tree with boosting
    * SVM
    * Random forest
* We run k-fold cross validation [k=10] to more robustly assess classifier accuracy beyond counting misclassified points
* We generate ROC curves/AUC for each classifier to provide another estimate of performance beyond accuracy 
* We check feature importance [when possible, not all models, like SVM with nonlinear kernel, allow] to better understand why our model performance is often not great 


## Here are the files included in this repo [note: we are not including MIMIC files as they are access-controlled]: 
* exploratory.Rmd
	* Sarah's R script that filters out unique patients based on whether or not they have text notes in MIMIC's NOTEEVENTS table
* SubID\_ICD\_Diseases.ipynb_ 
	* Saejeong's python notebook that finds subject ids for cardiomyopathy/coronary artery disease patients using MIMIC's D\_ICD\_DIAGNOSES and DIAGNOSES\_ICD tables
* CBB\_750\_final\_project.ipynb
	* A and Sarah's python notebook that extracts the most recent notes for each patient, runs doc2vec, generates labels and classifies/evaluates performance
* nontext
	* Jason's directory for non-text data matrix + related scripts, including the following files: 
		* nontext/mimic\_processing.Rmd
			* Jason's R script that grabs the most recent LABEVENTS data for each CM/CAD patient and formats into a data matrix
* Classifier\_visualization.Rmd
	* Saejeong's supplementary script that cleanly visualizes accuracy/k-fold accuracy/AUC across different tasks 
