# COMP 4630 Assignment 3 Report

**Team Name:** `[Duplicate]`

**Team Members:** `Tobias`, `Clency`, `Glenn`

## Abstract

This report details the development and evaluation of a deep learning model designed to classify Stack Overflow question titles into one of the top 10 most frequent single tags. The dataset was sourced from the BigQuery public Stack Overflow dataset. Our final model utilizes a hybrid approach, combining word-level embeddings with character-level features extracted via a Convolutional Neural Network (CNN). These combined features are processed by a Bidirectional Long Short-Term Memory (BiLSTM) network enhanced with an attention mechanism to focus on salient parts of the title. The model was trained using PyTorch, incorporating techniques like class weighting to handle data imbalance, AdamW optimization, cosine annealing learning rate scheduling, and early stopping. The final model achieved a test accuracy of **`~77%`** (see notbook). 

## Dataset Description

**Data Source & Selection:** The data consists of question titles and their associated tags, queried from the `bigquery-public-data.stackoverflow.posts_questions` table on Google BigQuery. To simplify the classification task, the query specifically selected questions that had *only one* tag associated with them. Furthermore, it restricted the selection to the 10 most frequent single tags within the dataset. A random sample of 120,000 such questions was retrieved to form the initial dataset.

**Preprocessing:** Titles underwent a cleaning process involving
 * Conversion to lowercase.
 *   Removal of special characters, preserving only letters, numbers, whitespace, plus signs (`+`), and hash symbols (`#`) which are common in programming contexts (for example: c++, #include).
 *   Lemmatization using NLTK's `WordNetLemmatizer` to reduce words to their base form.
 *   Removal of rows with empty titles after cleaning and deduplication based on the cleaned title.

**Biases and Limitations:**
 *   **Tag Scope:** The dataset is heavily biased towards only the top 10 most popular single tags (e.g., `javascript`, `java`, `c#`, `python`). It completely ignores less frequent tags and the vast majority of questions which have multiple tags. Therefore, the model's applicability is limited to this specific subset of tags.
 *   **Single Tag Focus:** Real-world Stack Overflow questions often have multiple tags (frameworks, languages etc. for example C++ UE5(unreal engine)). This dataset simplifies the problem to single-label classification, not reflecting the true "multi-tag" nature of the platform.
 *   **Title-Only:** The model only uses the question *title*. It lacks the context provided by the question body, code snippets, or user comments, which often contain crucial information for accurate tagging.
 *   **Temporal Snapshot:** The dataset represents a snapshot in time. Tag popularity and question phrasing evolve, potentially making the model less effective on newer data.
 *   **Language:** The data is all in english (at least what we have seen. We do know there is a spanish version of the stackoverflow site but not sure if its included in this data set)
 *   **Cleaning Artifacts:** While cleaning aims to standardize text, it might occasionally remove meaningful symbols or context. Lemmatization can also sometimes alter meaning slightly. So far this risk is generally minimal (at least they way we have implemented it) and worth the benifts 

**Class (im)balance:**
 *   As observed in the data exploration phase (value counts and bar plot), the dataset exhibits significant class imbalance. Tags like `javascript`, `java`, and `c#` were substantially more frequent than others in the top 10 list (e.g., `c++`, `ios`).
 *   This imbalance poses a challenge, as a naive model might become biased towards predicting the majority classes, achieving high overall accuracy but performing poorly on minority classes.
 *   We addressed this by calculating class weights (`sklearn.utils.class_weight.compute_class_weight`) based on the training set distribution and incorporating these weights into the `CrossEntropyLoss` function, penalizing misclassifications of minority classes more heavily.

**Impressions:**
*   The dataset provides a large corpus of relevant, domain-specific text (programming question titles).
*   The titles are generally short to medium length, making sequence models feasible. The `MAX_SEQ_LENGTH` of 40 seemed appropriate based on the title length distribution analysis, capturing a large majority of titles.
*   The single-tag restriction significantly simplifies the problem but also makes it somewhat artificial compared to the real Stack Overflow tagging task.
*   The inherent noise and variability in user-generated text means we need to be carefull with what we choose to do and how we do pre proccessing 
*   The class imbalance was a prominent feature that required explicit handling during model training.

## Experiment Log

Our development process involved iterative refinement of preprocessing, model architecture, and training strategy:

1.  **Preprocessing & Vectorization:**
    *   **Initial:** Basic lowercase and punctuation removal.
    *   **Refinement:** Added lemmatization (`WordNetLemmatizer`) to group word forms. Decided against stemming as lemmatization is less aggressive and preserves meaning better. Kept `+` and `#` as they are significant in tag names (c++, c#).
    *   **Vocabulary:** Built word vocabulary based on training data, using `MIN_WORD_FREQ=3` to filter out rare words (potential noise/typos). Introduced `<pad>` and `<unk>` tokens.
    *   **Character Vocab:** Created a character vocabulary including `<pad>` and `<unk>` to handle subword information.
    *   **Padding/Truncation:** Implemented fixed-length padding/truncation for both word sequences (`MAX_SEQ_LENGTH=40`) and characters within words (`MAX_WORD_LENGTH=15`) for batch processing. These values were chosen based on data exploration (histograms).

2.  **Model Architecture:**
    *   **Baseline (Conceptual):** Considered a standard word embedding -> LSTM -> Dense layer approach.
    *   **Enhancement 1 (Character Features):** Added a character embedding layer followed by a 1D CNN and max-pooling. The goal was to capture morphological features and handle out-of-vocabulary (OOV) words or typos effectively. The output character features were concatenated with word embeddings.
    *   **Enhancement 2 (Bidirectionality):** Changed the LSTM to a BiLSTM (`bidirectional=True`) to allow the model to learn context from both past and future words in the sequence. Hidden dimensions were adjusted accordingly.
    *   **Enhancement 3 (Attention):** Incorporated an attention mechanism (`AttentionLayer`) after the BiLSTM. This allows the model to weigh the importance of different words in the title when forming the final representation for classification, potentially improving performance on longer or more complex titles.
    *   **Regularization/Stabilization:** Added `Dropout` after embeddings, concatenated features, and before the final layer. Included `LayerNorm` after the first fully connected layer for potentially better gradient flow and stabilization.

3.  **Hyperparameters, Loss, Optimization:**
    *   **Embeddings:** Word (`WORD_EMBEDDING_DIM=300`) and Character (`CHAR_EMBED_DIM=50`, `CHAR_CNN_OUT_CHANNELS=100`) dimensions were chosen as common starting points.
    *   **LSTM:** `HIDDEN_DIM=256`, `NUM_LAYERS=2`. Multiple layers allow learning more complex patterns.
    *   **Loss Function:** `CrossEntropyLoss`. Critically, added `weight=class_weights` computed from the training set to counteract class imbalance.
    *   **Optimizer:** Switched from standard `Adam` to `AdamW` (`LEARNING_RATE=3e-4`, `WEIGHT_DECAY=1e-5`) which decouples weight decay from the gradient update, which seemed to better generalization.
    *   **Learning Rate Scheduling:** Implemented `CosineAnnealingLR` to gradually decrease the learning rate over epochs, helping the model converge more smoothly to a good minimum.
    *   **Gradient Clipping:** Used `nn.utils.clip_grad_norm_` with `max_norm=1.0` to prevent exploding gradients.
    *   **Training Control:** Employed `EarlyStopping` based on validation loss (`patience=5`) to prevent overfitting and terminate training when performance on the validation set ceased to improve. Saved the model weights corresponding to the best validation accuracy epoch.
    *   **Batch Size:** `BATCH_SIZE=32` based on the hardware we had this seemed to be a fair trade between performance and training time (2070super 8gb, colab t4 15gb?, 4060ti 16gb)

4.  **Learnings:**
    *   Combining word and character-level features significantly benefits text classification, especially with domain-specific terms, potential typos, or "out of vocab" words
    *   Attention mechanisms effectively improve performance by allowing the model to focus on relevant parts of the input sequence.
    *   Explicitly addressing class imbalance (using class weights) is crucial for fair performance across all classes, not just the majority ones.
    *   ptimizers (like AdamW), learning rate scheduling (Cosine Annealing), and regularization techniques (Dropout, LayerNorm, Gradient Clipping, Weight Decay) are super important/useful
    *   Early stopping based on a validation set is vital to prevent overfitting and reduce unnecessary training time.

## Discussion/Conclusion

**Best Model:** Our best performing model is the `EnhancedTextModel` described in the code. It integrates:
 *   Word embeddings (300 dim).
 *   Character embeddings (50 dim) processed by a 1D CNN (100 output channels, kernel size 3).
 *   Concatenation of word and character features for each token.
 *   A 2-layer Bidirectional LSTM (256 hidden dim per direction).
 *   An Attention mechanism applied to the BiLSTM output.
 *   Fully connected layers with ReLU activation, Dropout (0.5), and Layer Normalization, leading to the final classification output (10 classes).
 This model achieved **`~77%`** accuracy on the held out test set.

**Challenges, Advantages, and Limitations:**
 *   **Challenges:**
     *   Computational resources: Training deep models, especially LSTMs, can be time-consuming and GPU memory-intensive.
     *   Hyperparameter tuning: Finding the optimal combination of dimensions, layers, dropout rates, learning rates, etc., requires experimentation.
     *   Preprocessing decisions: Balancing text cleaning with preserving meaningful information (like `c++` vs `c`).
     *   Handling class imbalance effectively.
 *   **Advantages:**
     *   Hybrid approach (word + char): Robust to oput of vocab words and captures subword patterns.
     *   BiLSTM: Captures sequential context effectively.
     *   Attention: Improves performance by focusing on important words and offers potential for interpretability (though not explored here).
     *   Robust training: Use of class weights, AdamW, LR scheduling, and early stopping leads to better generalization and efficiency.
 *   **Limitations:**
     *   As discussed (Dataset section), limited to single, top-10 tags and relies solely on title text.
     *   The model complexity might be overkill for some simpler titles but necessary for others.
     *   Fixed sequence length (`MAX_SEQ_LENGTH`) means very long titles are truncated.

**Key Takeaways:**
 *   Deep learning models, particularly hybrid architectures combining different feature extraction methods (word, character) and sequence processing (BiLSTM, Attention), from what we read online during this asgg are highly effective for text classification tasks.
 *   Careful data preprocessing and understanding dataset characteristics (like class imbalance) are critical first steps.
 *   A systematic approach to training, including appropriate optimization, regularization, learning rate scheduling, and early stopping, is essential for achieving good performance and preventing overfitting.
 *   The specific choices made (lemmatization, character CNN, attention, class weights) seem to all directly impact performance noticably. 

**If we had more time:**
 *   **Multi-Label Classification:** Adapt the model to handle questions with multiple tags (example: using a sigmoid activation in the final layer and binary cross-entropy loss).
 *   **Incorporate Question Body:** Extend the input to include the question body text, potentially using hierarchical attention models or transformers.
 *   **Pre-trained Embeddings:** Initialize word embeddings with pre-trained vectors like GloVe or FastText, potentially fine-tuning them.
 *   **Transformer Models:** Experiment with transformer-based architectures (like BERT or its variants), which are state-of-the-art for many NLP tasks, fine-tuning a pre-trained model on this dataset.
 *   **Hyperparameter Optimization:** Use automated tools (e.g., Optuna, Ray Tune) for more extensive hyperparameter searching.
 *   **Error Analysis:** Perform a detailed analysis of misclassified examples to understand model weaknesses and guide further improvements.
 *   **Attention Visualization:** Visualize attention weights to interpret which words the model focuses on for prediction (if this is even possible, as in is there a way to visualize it such that it makes sense?)

## Reflections

Our primary challenge during this assignment revolved less around the intricacies of the model architecture itself and more around the initial infrastructure setup. Configuring Google Colab, establishing the connection to BigQuery, and particularly navigating Google credential management within a collaborative team setting proved to be the most significant hurdles. On the model development side, a key takeaway, perhaps more apparent than in previous assignments, was the tangible relationship between data scale, training time, and achievable performance. We consistently observed that with our current dataset size (approximately 120,000 cleaned titles focused on the top 10 single tags), the model's test accuracy seemed to plateau around the 75-81% mark. This experience suggests that while our architecture effectively captured many patterns within this data scope, breaking significantly beyond this performance ceiling would likely necessitate a substantial expansion of the training dataset and, consequently, considerably more extensive training time and computational resources. It reinforced the idea that substantial gains in accuracy often require proportionally larger investments in data and training effort.

Our ah-hah moments were probably incorporating the character CNN yielded a noticeable improvement in validation accuracy, highlighting the value of subword information for this task. Similarly, applying class weights didn't always drastically increase overall accuracy, but it significantly boosted performance on minority classes, resulting in a more balanced and fair classifier, as evidenced by per-class metrics and the confusion matrix. Finally, it was somewhat surprising how effectively the model performed relying solely on question titles, suggesting that titles alone often carry strong signals for identifying the primary topic or tag on Stack Overflow.

## Appendices

### References

* PyTorch Documentation: https://pytorch.org/docs/stable/index.html
* NLTK Documentation: https://www.nltk.org/
* Scikit-learn Documentation: https://scikit-learn.org/stable/documentation.html
* Google Cloud BigQuery Python Client Library: https://cloud.google.com/python/docs/reference/bigquery/latest

### Loading code

To load the saved model, vocabulary, and configuration (in the notebook) after running the notebook

```python
model = torch.load(model_path, weights_only=False) 
```
to run the saved model in the repo please use inference.py
