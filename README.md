# Milestone 1 Report

## Our Goal

For this milestone we decided to tailor the cleaning and pre-processing pipelines for a **topic modelling** task.

We started off by carrying out various analysis tasks to gain insight on how the data was originally formatted, its content, certain relationships and much more. We then used this analysis to guide our cleaning and pre-processing stages. The rest of this report will be talking about each stage we carried out, the reasoning behind each step and why we decided to do it that specific way.

## Data

For our project, we used data from the Da7ee7 YouTube channel. We acknowledge and credit Malak Labib and Hamza Gehad. for curating and providing access to this dataset. https://drive.google.com/drive/u/0/folders/10SnpI_Z6sxbAT1b1NXvL8Q7iLzEnJ0oO 

## Analysis

### Document Length Distribution Analysis

- Method: We compute and visualize the number of words per document.
- Visualization: A histogram showing document length distribution.
- Reasoning: Ensures that most documents are sufficiently long for meaningful topic extraction. Detects outliers.

![Document Length Distribution](./images/document-length.png)

### Bigram Frequency Analysis

While unigram (single-word) analysis gives us insights into the most frequent words, bigram analysis reveals common word pairs, which can:

Capture phrases that provide richer topic meaning (e.g., "climate change" instead of "climate" and "change" separately).
Improve topic coherence, as certain concepts are better represented as pairs rather than single words.

![Bigram Frequency](./images/bigrams.png)

### Word Frequency Analysis

- Method: Count occurrences of words across documents.
- Visualization: A word frequency histogram or a word cloud.
- Reasoning: Identifies most common words in the corpus. Helps determine if stopwords were effectively removed.

![Word Cloud](./images/cloud.png)

### TF-IDF Analysis

- Method: Compute TF-IDF scores to determine important words per document.
- Reasoning: High TF-IDF scores indicate document-specific importance. Helps visualize which words differentiate topics rather than just appear frequently.

## Cleaning

For cleaning our data we:

- Remove the timestamps.
- Remove punctuations and numbers.
- Remove single letters.
- Remove repeated words.

The combination of these cleaning steps ensures that our dataset is free from unnecessary noise, improving the quality of the text data for further analysis. By removing timestamps, we eliminate nuances that may not be relevant to our task. Removing punctuation and numbers helps standardize the text, making it easier to process. Single letters, often artifacts of tokenization or errors, are removed to prevent misinterpretation. Lastly, eliminating repeated words reduces redundancy and overall helps by keeping the documents consistent.

These steps enhance the dataset's readability and usability, making it more suitable for our later processing stages.

## Pre-processing

### 1. Tokenization

- Method: We use RegexpTokenizer to split the text into individual words while preserving punctuation when necessary.

- Reasoning: Tokenization is an essential step before feature extraction. It allows us to work with individual cleaned words while preserving context. Regular expressions provide more flexibility compared to traditional whitespace-based tokenization.

### 2. Stopword Removal

- Method: We use a custom list of stopwords we generate from our analysis stage and remove those stopwords from the tokenized words.
- Resoning: Stopwords do not contribute to topic differentiation and by removing them we would be reducing the vocabulary size and improving topic separation. This will leave behind meaningful words that define topics, rather than generic fillers.

### 3. Stemming

- Method: We use the ISRIStemmer, which is optimized for Arabic text processing.

- Why Stemming and Not Lemmatization?
  Stemming reduces words to their root form by removing prefixes/suffixes (e.g., "كتابة" → "كتب").
  Lemmatization maps words to their dictionary base form (e.g., "better" → "good" in English).
  Lemmatization is often more precise, but it requires linguistic rules and a morphological analyzer.
  In Arabic NLP, stemming is preferred due to the complex nature of Arabic morphology. Lemmatization is computationally expensive and requires additional resources.
  Example:
  "مدرسة" (school) → "درس" (study/lesson) after stemming.

- Reasoning: This reduces dimensionality = reducing computational complexity. Moreover, the model will become more generalizable since word variants will now be mapped to the same word and the sparsity of data will be reduced.

### 4. TF-IDF

After cleaning, we need to convert text into numerical form for topic modeling.

- Method: We use TfidfVectorizer from sklearn to transform the text corpus into a TF-IDF matrix.

- Reasoning: Bag-of-Words (BoW) represents documents using raw word counts. However, frequent words dominate, reducing topic differentiation. TF-IDF (Term Frequency-Inverse Document Frequency) assigns lower weights to frequent words and higher weights to rare, meaningful words.

### 5. Named Entity Recognition (NER)

Although topic modeling focuses on general themes, Named Entity Recognition (NER) can provide additional insights.

- Method: We use a named entity recognition (NER) model to extract entities (e.g., names, organizations, locations).

- Reasoning: Certain named entities may be highly relevant to topics.

NER can help refine topic labels by identifying key figures, places, or institutions. Example Use Case: If an LDA model identifies a topic related to technology, NER can extract specific company names (e.g., "Apple", "Google"), making the topic labels more interpretable.
