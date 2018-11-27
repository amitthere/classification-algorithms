# Classification Algorithms

3 classification algorithms implemented and tested on 4 datasets.

Performance of all classifiers is validated using **10-fold Cross Validation**.

Accuracy, Precision, Recall, and F-1 measure of the classifiers is reported.

## 1. Nearest Neighbour Classifier

## 2. Decision Tree Classifier

### * Random Forest Classifier based on the Decision Tree Classifier

### * Boosting based on the Decision Tree Classifier

## 3. Naïve Bayes Classifier


### Test Datasets

|     Dataset       |  Objects  | Number of Classes |
|:-----------------:|:---------:|:-----------------:|
| dataset1          |    569    |        2          |
| dataset2          |    462    |        2          |
| project3_dataset3 |    100    |        2          |
| project3_dataset4 |    14     |        2          |


### Dataset Format

Each row represents a gene:
1. Each line represents an object
2. Last column is the class label
3. Rest of the columns represent feature values, each of them can be a real-value (continuous type) or a string (nominal type)
