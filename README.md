# post-helpfulness: NAACL 2019

Sources for our paper titled "Predicting Helpful Posts in Open-Ended Discussion Forums: A Neural Architecture", NAACL 2019.

If you use these sources, please consider citing the paper:

```
@inproceedings{halder-etal-2019-predicting,
    title = "Predicting Helpful Posts in Open-Ended Discussion Forums: A Neural Architecture",
    author = "Halder, Kishaloy and Kan, Min-Yen and Sugiyama, Kazunari",
    booktitle = "Proceedings of the 2019 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)",
    month = jun,
    year = "2019",
    pages = "3148--3157",
```

## Files
* README.md: This file.
* relevance_novelty.py: Main file to run the code.
* utilities.py: Contains some utility functions.
* constant.py: Contains values for some parameters.

## Instructions to run
1) It requires pre-trained glove embedding file. It can be downloaded from here. https://nlp.stanford.edu/projects/glove/
Once you have extracted the `.zip` file, use the full path of the embedding file (named `glove.6B.100d.txt`) in `constants.py` as the value of the `WORD_EMBEDDING_FILE` variable.

2) You also need to provide path of the data file as the value of `POSTS_FILE` in `constants.py`. This file should be a tab (`\t`) separated file having the following data columns:
* `postID`: A unique ID for each post.
* `postText`: Textual content of the target post to be classified.
* `OPost`: Textual content of the original post in the same thread as that of the target post.
* `helpfulCount`: A binary value (0 or 1) depicting the two classes (non-helpful and helpful respectively).
* `context_[0-19]`: 20 separate columns each with a contextual post from the same thread as that of the target post. Note that the `19th` (`context_19`) column would always be the penultimate post in the same thread. As for example, if the target post is the `5th` in the thread, then `context_14` would be the first post in the thread, `context-15` the second, similarly `context_19` would be the fourth (and thus penultimate) in the thread. Other columns (`context_[0-13]`) would be left empty.

Once you have these in place, you can train the model using the following:

```
python3 relevance_novelty.py
```
All the outputs will be logged in a file in the `./logs` folder.

## Requirements
You would need the following installed in your system to run the sources.
* Python 3.x
* keras
* pandas
* sklearn
* tqdm
