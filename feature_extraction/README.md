This folder contains the feature extraction pipeline for Brain4Cars. You need to download all the param files from the Brain4Cars dataset before you can use this code.

###About the files:
extractFeatures: The core feature extraction code. Generates angular histograms of landmark/KLT points. Works with both CLM and KLT. Tested only on CLM right now.

bulkFeatureExtract: Used for extracting features for baseline non-temporal models like SVMs, Random Forests, Logistic Regression.

bulkFeatureExtractCLM: CLM/CCNF version of the above.