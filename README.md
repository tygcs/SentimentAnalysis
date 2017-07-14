# SentimentAnalysis


### process_data.py

Deal with corpus data including reading the hotel comments data, the words dicts for naive_SA ...


### naive_SA.py

Use dictionaries to predict the sentiment(1, 0, -1) of the given text.

Run:

    python ./naive_SA.py


### bayes.py

Use naive bayes to classify the text.

Run:

    python ./bayes.py


### textcnn

Use CNN to do text classification.

Run `textcnn.py` to do `train(train_data_path)` first, the model will be saved in ./textcnn/model/

Then run `textcnn.py` to do `test(test_data_path, r'model/xxx/checkpoints')`.