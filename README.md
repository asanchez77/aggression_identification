# aggression_identification
Identification of aggresive language using Logistic Regression classifier.

**Best parameters for the multinomial Logistic Regression classifier:** 
  * penalty = 'l2',
  * multi_class = 'multinomial' ,
  * solver='lbfgs',
  * C= 5.0,
  * max_iter = 300

**Best parameters for the ovr Logistic Regression classifier:** 
  * penalty = 'l2',
  * multi_class = 'ovr' ,
  * solver='liblinear',
  * C= 10.0,

## agg_class_log_reg_multinomial.py 
Used to classify agr_en_dev.csv using the Logistic Regression classifier using the multinomial scheme.

## agg_class_log_reg_ovr.py

Used to classify agr_en_dev.csv using the Logistic Regression classifier using one-versus-rest scheme.
