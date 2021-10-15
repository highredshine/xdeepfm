https://www.kaggle.com/mrkmakr/criteo-dataset/code
https://www.kaggle.com/c/criteo-display-ad-challenge/code



This dataset contains feature values and click feedback for millions of display 
ads. Its purpose is to benchmark algorithms for clickthrough rate (CTR) prediction.
It has been used for the Display Advertising Challenge hosted by Kaggle:
https://www.kaggle.com/c/criteo-display-ad-challenge/

===================================================

Full description:

This dataset contains 2 files:
  train.txt
  test.txt
corresponding to the training and test parts of the data. 

====================================================

Dataset construction:

The training dataset consists of a portion of Criteo's traffic over a period
of 7 days. Each row corresponds to a display ad served by Criteo and the first
column is indicates whether this ad has been clicked or not.
The positive (clicked) and negatives (non-clicked) examples have both been
subsampled (but at different rates) in order to reduce the dataset size.

There are 13 features taking integer values (mostly count features) and 26
categorical features. The values of the categorical features have been hashed
onto 32 bits for anonymization purposes. 
The semantic of these features is undisclosed. Some features may have missing values.

The rows are chronologically ordered.

The test set is computed in the same way as the training set but it 
corresponds to events on the day following the training period. 
The first column (label) has been removed.

====================================================

Format:

The columns are tab separeted with the following schema:
<label> <integer feature 1> ... <integer feature 13> <categorical feature 1> ... <categorical feature 26>

When a value is missing, the field is just empty.
There is no label field in the test set.

====================================================

Dataset assembled by Olivier Chapelle (o.chapelle@criteo.com)

The data files for this competition are no longer available here. Please visit this link to access the files.
File descriptions
train.csv - The training set consists of a portion of Criteo's traffic over a period of 7 days. Each row corresponds to a display ad served by Criteo. Positive (clicked) and negatives (non-clicked) examples have both been subsampled at different rates in order to reduce the dataset size. The examples are chronologically ordered.
test.csv - The test set is computed in the same way as the training set but for events on the day following the training period.
random_submission.csv - A sample submission file in the correct format.
Data fields
Label - Target variable that indicates if an ad was clicked (1) or not (0).
I1-I13 - A total of 13 columns of integer features (mostly count features).
C1-C26 - A total of 26 columns of categorical features. The values of these features have been hashed onto 32 bits for anonymization purposes. 
The semantic of the features is undisclosed.

When a value is missing, the field is empty.