<h1 align="center">
  <strong>The Intelligent Conversationalist</strong>
  <span style="font-size: 14px; font-weight: normal;"> (Graduation Project)</span>
</h1>


## Video Illustrates how the Project Work

<video src="https://github.com/user-attachments/assets/9c3aaf5f-f70e-4a42-b009-c6b95208cb16" width="400" controls></video>

<br><br>


## Table That Has Every File Code with its Description

| File Name                              | Description |
|----------------------------------------|-------------|
| **detect_languages**                   | Detects language in JSON files from *Accepted_Islam*, *Interested_in_Islam*, and *Wants_to_Convert* using **`langdetect`**. Organizes files into language-based subdirectories, counts occurrences, and saves with sequential names (e.g., `Accepted_Islam1.json`). |
| **rename_files**                       | Renames JSON files in *Accepted_Islam*, *Interested_in_Islam*, and *Wants_to_Convert* using **`natsorted`** for correct order. Sequential naming convention (e.g., `Accepted_Islam1.json`). |
| **choosing_features_and_removing_unwanted_convs** | Filters JSON conversations by removing those with fewer than 5 messages or visitor name = "Visitor". Structures data into **Visitor/Daei** messages and attaches ratings if available. Saves as sequential JSONs. |
| **add_convLength_avgMsgLength_#ofMessages** | Adds features: **conversation_length (total words)**, **average_message_length**, and **number_of_messages** to JSON conversations. Preserves structure and saves updated files into new root folders. |
| **add_sentimentScoreFeature**          | Uses a **RoBERTa (Twitter-specific, pre-trained) sentiment model** to compute scores per message. Adds **answers_average_score (Daei)**, **questions_average_score (Visitor)**, and **conversation_average_score**. Supports **GPU acceleration**. |
| **split_seed_train&valid** *(not used)* | Splits conversations into **training/validation** ensuring both participants (Visitor/Daei) are present in the first split (≥5 messages). Renames files sequentially. |
| **expand_convos_50%andmore**           | Splits conversations into multiple percentages (50%, 60%, 70%, etc.). Updates stats such as **sentiment scores** and **message lengths**, and saves focused subsets for training. |
| **Experiment2_model**                  | **TensorFlow** deep learning model combining text + features. Text preprocessing (lowercasing, non-alphanumeric removal, stopword removal). Uses **pre-trained GloVe embeddings**, **CNN + BiLSTM + multi-head attention**, and additional features (e.g., conversation length, message counts). Handles class imbalance with **sklearn MinMaxScaler + class weights**. Trained with **Adam optimizer**, early stopping, learning rate reduction. Saves model + tokenizer + scaler. |
| **Expirement3_Dataset_Preparation**    | Prepares dataset by selecting **Daei answers** with ≥5 preceding questions and ≥1 following. Computes categories and probabilities via **Experiment2_model**. Defines target labels (Positive/Negative/Neutral) based on category probability shifts (threshold = 0.1). Combines consecutive Daei answers and averages **sentiment scores**. |
| **Experiment3_model (First Approach)** | **Deep learning classifier** predicting Positive vs Negative. Inputs = **Daei message text** + **sentiment score**. Neutral labels excluded to make the problem more abstract and therefore improving results. |
| **Experiment3_model_V2**               | **BERT (bert-base-uncased)** classifier fine-tuned on **Daei messages only**. Preprocessing = lowercase + cleaning. Tokenized to **256 tokens**, split 70/15/15. Includes **class weights**, **dropout dense layer (64 units)**, trained with **Adam (3×10⁻⁵)**, batch size 16, early stopping. Saves **TensorFlow SavedModel + tokenizer**. Evaluated with **precision/recall/F1**. |
| **Experiment3_model_V2.1**             | Same as V2 but excludes **Wants_to_Convert** dataset. Adds **learning-rate scheduler**: starts at 3×10⁻⁵, decays 10%/epoch after epoch 10. |
| **Experiment3_model_V2.2**             | Same as V2 but trains only on **Accepted_Islam** data. Uses **ReduceLROnPlateau** (reduce LR by 10% when validation loss stalls). Retains **bert-base-uncased** architecture, class weighting, early stopping, and TensorFlow pipeline. |

<br>

## Dataset Used For Training and How We Splitted it
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/dbe3e6d6-da6c-43a8-9765-abb4f132dbb6" />

<br><br>
## Experiment 1 (add_sentimentScoreFeature)
<img width="1816" height="1049" alt="image" src="https://github.com/user-attachments/assets/513e3a6d-62bd-471f-b6e9-cb3477d3081c" />
<br><br><br>


## Fine-Tuning Results
### Experiment 2
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/c58526bf-472c-4c12-82e2-8f00871a3d52" />
<br><br>

### Experiment 3 V2.2
<img width="1000" height="600" alt="image" src="https://github.com/user-attachments/assets/4f21c4a8-9264-41ba-a36f-6c4d68a94ce9" />
