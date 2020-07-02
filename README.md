Trends in Online Dating
==============================
For this project, I utilized NLP techniques to understand how OkCupid profiles changed with age, sex, and pet preference! The motivation was largely personal; I was about to turn 30 (!!) and I was single, so I was curious to see how the dating landscape might abruptly change. In addition, I love online dating, as I find it fascinating to see how people represent themselves and what they find most 'marketable' about themselves. Lastly, with there being so many dating services these days, I thought it'd be interesting to do some customer segmentation and understand what drove the demand for so many services.

I performed some data cleaning to get rid of links, html tags, and other oddities, tried a few types of vectorizers, and performed some topic modeling via NMF (Non-negative Matrix Factorization). I also borrowed a concept called the scaled F-score from [Scattertext](https://github.com/JasonKessler/scattertext) to see which terms were most common and unique within a given category. Between these two techniques, I was able to understand how dating needs, self-representation, and values changed with age, sex, and, to an extent, pet preferences!

For detailed results, [read my blog post](https://elarson649.github.io/2020/05/27/okcupid/) or check out the presentation in the repo!

Project Organization
------------

**Data**
  * Raw:
    * profiles.csv.zip: The dataset with 60,000+ OkCupid profiles


**Notebooks**
  * Project_4_final.ipynb: Notebook that walks through the cleaning, engineering, modeling, and visualization process. References functions in /src

**Src**
  * data:
    * make_dataset.py: Gets data from profiles.csv/MongoDB, performs some cleaning, and undersamples so we have balanced classes
  * features:
    * cleaning_data.py: Cleans text data- lots of regex, part-of-speech tagging, lemmatization, and tokenization
    * vectorize_data.py: Vectorizes the data into count and TFIDF vectorizers
  * models:
    * topic_model.py: Performs dimensionality reduction and displays the top topics. I tried NMF, LSA, and LDA, but went with NMF in the end
    * unique_terms.py: Finds the scaled f-score as described above and [from the creator of Scattertext](https://github.com/JasonKessler/scattertext#understanding-scaled-f-score)
  * visualization:
    * bar_graph.py: Creates some colorful bar graphs!
--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
