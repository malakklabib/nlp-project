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

As we can see there are some outliers in the document lengths. Removing them before the machine learning task would be a good decision.

### Bigram Frequency Analysis

While unigram (single-word) analysis gives us insights into the most frequent words, bigram analysis reveals common word pairs, which can:

Capture phrases that provide richer topic meaning (e.g., "climate change" instead of "climate" and "change" separately).
Improve topic coherence, as certain concepts are better represented as pairs rather than single words.

![Bigram Frequency](./images/bigrams.png)

Using these Bigrams as tokens would capture more meaning and contribute to a more efficient topic modeling pipeline.

### Word Frequency Analysis

- Method: Count occurrences of words across documents.
- Visualization: A word frequency histogram or a word cloud.
- Reasoning: Identifies most common words in the corpus. Helps determine if stopwords were effectively removed.

![Word Cloud](./images/cloud.png)

From this word cloud we gain insights on which words we can remove as stopwords and so on.

### TF-IDF Analysis

- Method: Compute TF-IDF scores to determine important words per document.
- Reasoning: High TF-IDF scores indicate document-specific importance. Helps visualize which words differentiate topics rather than just appear frequently.

### Sentiment Analysis

- Model: Ammar-alhaj-ali/arabic-MARBERT-sentiment model from Hugging Face's Transformers library. 
- Text Processing: Each document was processed in chunks of 150 words to handle long text effectively. Each chunk was classified into one of three categories: negative, neutral, or positive.
- Results Aggregation: Sentiment classification was aggregated for each document by calculating the average probabilities of all chunks and assigning the sentiment with the highest count as the final sentiment label.

![Sentiment Analysis](./images/sentiment_analysis.png)

#### Category vs. Average Engagement

- **Method:** We computed the average engagement (views, likes, and comments) per category.
- **Visualization:** Horizontal bar plot showing engagement levels for each category.
- **Reasoning:** Identifies which content categories drive the most audience interaction.

![Category vs Average Engagement](./images/category-average.png)

**Insight:**

- History & Politics had the **highest average engagement**, while Health had the lowest.
- Categories like **Science & Technology** and **Religion** also performed well, indicating public interest in these subjects.

#### **Most Common Entities for a Label (Example: Politics)**

- **Method:** Extracted the top entities appearing under the **Politics** label using NER.
- **Visualization:** Horizontal bar chart displaying the **top 5 most frequently mentioned entities**.
- **Reasoning:** Identifies the most commonly discussed names, places, or organizations within a category.

![Top Entities in Politics](./images/entity-politics.png)

**Insight:**

- "روميل" (Rommel), "هتلر" (Hitler), and **major geopolitical regions** were dominant in political discussions.
- This suggests that historical political figures and past events are highly referenced in this category.

#### **Most Common Entities in Viral Content**

- **Method:** Identified the most frequently appearing named entities in **viral content (1M+ views).**
- **Visualization:** Bar chart ranking the top 10 most frequent entities.
- **Reasoning:** Helps determine **which topics contribute to virality**.

![Top Entities in Viral Content](./images/entity-viral.png)

**Insight:**

- "مصر" (Egypt), "الهند" (India), and "الصين" (China) were among the most frequent entities in viral content.
- This suggests that **historical and geopolitical topics** play a strong role in engagement.

#### **Entity Co-Occurrence Network Analysis**

- **Method:** Created a network graph showing which entities frequently appear together in viral content.
- **Visualization:** A network graph of entity co-occurrence relationships.
- **Reasoning:** Helps identify potential thematic connections between entities.

![Entity Co-Occurrence Network](./images/co-occurence.png)

**Insight:**

- The network did not reveal strong insights, as many entities appeared loosely connected rather than forming meaningful clusters.
- **Negative results are still valuable**, as they indicate that entity relationships in viral content may be more context-dependent rather than rigidly structured.

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

### Limitations of Our Framework

While our cleaning and preprocessing pipeline is well-optimized for topic modeling, there are certain limitations that should be acknowledged:

1. Sensitivity to Preprocessing Choices
   Certain preprocessing steps, such as stemming, may lead to loss of meaning for some words, affecting topic coherence.
   Stopword removal may unintentionally filter out words that hold contextual importance.
   Decisions like punctuation removal might impact phrase-based topic modeling approaches (e.g., bi-grams, tri-grams).
   Potential Improvement: A more dynamic approach where preprocessing choices are evaluated against topic coherence scores.

2. Loss of Context in Topic Modeling
   Bag-of-Words (BoW) and TF-IDF representations ignore word order, which may lead to loss of syntactic and semantic structure.

3. Most of the sentiment analysis results were classified as negative, which could indicate a limitation in the model's accuracy or bias in the dataset. The arabic-MARBERT model may not generalize well to specific contexts like this dataset's content. Further fine-tuning of the model might be necessary to improve accuracy.
