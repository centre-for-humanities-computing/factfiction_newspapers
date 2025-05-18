## Fact from Fiction

This repo accompanies our paper to distinguish feuilleton fiction in Danish newspapers. 

📝 In ```notes```you will find the annotation scheme for the fiction/nonfiction categorization 


### 🧭 Directions for code:
In ```scripts``` you'll find the code, including:
- ```get_features.py```to get MFWs, TF-IDF, and stylistic/syntactic/affective features, the functions of which are defined in ```scripts/feature_utils.py```.
- ```classify.py``` which employs a random forest model across our 4 different feature sets (MFW100, TF-IDF, selected features, and embeddings)
- ```descriptives.py``` which visualizes and test differences between the classes of fiction/nonfiction
- a ```clustering_task.py``` which tests embeddings for clustering feuilleton series (note that these need to be precomputed and are not available here because of size-issues)


Note that the script for creating embeddings (various) is at [this anonymized repo](https://anonymous.4open.science/r/encode_feuilletons-6922)

And that the script to benchmark SA models on the Fiction4 corpus is in [this anonymized repo](https://anonymous.4open.science/r/literary_sentiment_benchmarking-CF00)