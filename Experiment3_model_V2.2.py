# In this version we are training the model without Daei's sentiment score and without 'Wants to Covert' and
# 'Interested in Islam' experiment 3 data

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from transformers import BertTokenizer, TFBertModel
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
from sklearn.metrics import classification_report

# GPU Selection
def select_gpu(gpu_index):
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Restrict TensorFlow to only use the selected GPU
            tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')
            # Enable dynamic memory allocation
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            print(f"Using GPU: {tf.config.experimental.get_device_details(gpus[gpu_index])['device_name']}")
        except RuntimeError as e:
            print(f"Error configuring GPU: {e}")
    else:
        print("No GPU available. Running on CPU.")


# Choose GPU (set index of GPU to use, e.g., 0 for the first GPU)
gpu_index = 0
select_gpu(gpu_index)

# Confirm if GPU is being used
if tf.config.list_logical_devices('GPU'):
    print("GPU is being used.")
else:
    print("GPU is NOT being used. Running on CPU.")

# Preprocess Text
def preprocess_text(text):
    return text.lower().strip()

# Load Data
def load_data_from_folder(folder_path):
    data = []
    for file_path in glob(os.path.join(folder_path, '*.json')):
        with open(file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)

            prob_before = json_data.get('Probability_before')
            prob_after = json_data.get('Probability_after')

            if prob_before == 0 and prob_after == 0:
                target = 'Negative'
            elif prob_before == 1 and prob_after == 1:
                target = 'Positive'
            else:
                target = json_data.get('target')

            if target == 'Neutral':
                continue

            daei_message = json_data.get('Daei_message', "")

            if daei_message:
                data.append({
                    'Daei_message': daei_message,
                    'target': target
                })
    return data

# Path to data folder (only Accepted Islam data is used)
accepted_islam_path = r'D:\Roken Alhewar Datasset\filtered_results1\Accepted_Islam'

# Load data
data_accepted = load_data_from_folder(accepted_islam_path)

# Use only Accepted Islam data
data = data_accepted
df = pd.DataFrame(data)
df['Daei_message'] = df['Daei_message'].astype(str).apply(preprocess_text)

# Label encode targets
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])
print("Target classes:", label_encoder.classes_)

# Drop missing values (only check for messages now)
df = df.dropna(subset=['Daei_message'])
df = df.reset_index(drop=True)

# Initialize BERT tokenizer
bert_model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(bert_model_name)

# Tokenize text data
def encode_texts(texts, tokenizer, max_length):
    return tokenizer(
        texts.tolist(),
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='tf'
    )

MAX_SEQUENCE_LENGTH = 256
encoded_inputs = encode_texts(df['Daei_message'], tokenizer, MAX_SEQUENCE_LENGTH)
X_text = encoded_inputs['input_ids']
X_mask = encoded_inputs['attention_mask']

# Convert TensorFlow tensors to numpy arrays
X_text_np = X_text.numpy()
X_mask_np = X_mask.numpy()
y = df['target'].values

# Perform train-test split
X_text_temp, X_text_test, X_mask_temp, X_mask_test, y_temp, y_test = train_test_split(
    X_text_np, X_mask_np, y, test_size=0.15, random_state=42
)

X_text_train, X_text_val, X_mask_train, X_mask_val, y_train, y_val = train_test_split(
    X_text_temp, X_mask_temp, y_temp, test_size=0.1765, random_state=42
)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# Load pretrained BERT model
bert_model = TFBertModel.from_pretrained(bert_model_name)

# Build model (only text inputs)
text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='text_input')
mask_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='mask_input')
bert_output = bert_model(text_input, attention_mask=mask_input)[1]

fc = Dense(64, activation='relu')(bert_output)
fc = Dropout(0.3)(fc)
output = Dense(len(label_encoder.classes_), activation='softmax')(fc)

model = Model(inputs=[text_input, mask_input], outputs=output)

# Compile model with initial constant learning rate
initial_lr = 3e-5
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define performance-based learning rate scheduler using ReduceLROnPlateau
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=2, verbose=1, min_lr=1e-7)

# Train model
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices(({
    'text_input': X_text_train, 'mask_input': X_mask_train
}, y_train)).shuffle(1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices(({
    'text_input': X_text_val, 'mask_input': X_mask_val
}, y_val)).batch(batch_size)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
with tf.device(f'/GPU:{gpu_index}'):  # Use the selected GPU
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=25,
        callbacks=[early_stopping, lr_scheduler],
        class_weight=class_weights_dict,
        verbose=1
    )

# Save Model and Tokenizer
save_path = r'D:\Users\Research chair\Desktop\Roken Alhewar Datasset\Experiment3Model'
model.save(save_path)
print(f"Model saved at: {save_path}")

# Save Tokenizer
tokenizer_path = os.path.join(save_path, 'tokenizer.pickle')
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Tokenizer saved at: {tokenizer_path}")

# Evaluate and generate classification report
test_dataset = tf.data.Dataset.from_tensor_slices(({
    'text_input': X_text_test, 'mask_input': X_mask_test
}, y_test)).batch(batch_size)
with tf.device(f'/GPU:{gpu_index}'):  # Use the selected GPU
    y_pred = model.predict(test_dataset)
    y_pred_classes = tf.argmax(y_pred, axis=1).numpy()

report = classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_, digits=4)
print(report)

# Plot training history
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(history.history['loss'], label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[1].plot(history.history['accuracy'], label='Train Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Accuracy over Epochs')
    axes[1].set_xlabel('Epochs')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    plt.tight_layout()
    plt.show()

plot_training_history(history)
