# Import Libraries
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
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping

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
            daei_sentiment_score = json_data.get('Daei_sentiment_score', None)

            if daei_message and daei_sentiment_score is not None:
                data.append({
                    'Daei_message': daei_message,
                    'Daei_sentiment_score': daei_sentiment_score,
                    'target': target
                })
    return data

# Paths to data folders
accepted_islam_path = r'C:\Users\Research chair\Desktop\Roken Alhewar Datasset\filtered_results1\Accepted_Islam'
interested_in_islam_path = r'C:\Users\Research chair\Desktop\Roken Alhewar Datasset\filtered_results1\Interested_in_Islam'
wants_to_convert_path = r'C:\Users\Research chair\Desktop\Roken Alhewar Datasset\filtered_results1\Wants_to_Convert'

# Load data
data_accepted = load_data_from_folder(accepted_islam_path)
data_interested = load_data_from_folder(interested_in_islam_path)
data_convert = load_data_from_folder(wants_to_convert_path)

# Combine and preprocess data
data = data_accepted + data_interested + data_convert
df = pd.DataFrame(data)
df['Daei_message'] = df['Daei_message'].astype(str).apply(preprocess_text)

# Label encode targets
label_encoder = LabelEncoder()
df['target'] = label_encoder.fit_transform(df['target'])
print("Target classes:", label_encoder.classes_)

# Drop missing values
df = df.dropna(subset=['Daei_message', 'Daei_sentiment_score'])
df['Daei_sentiment_score'] = df['Daei_sentiment_score'].astype(float)
df = df.reset_index(drop=True)

# Normalize numerical features
X_num = df[['Daei_sentiment_score']].values
scaler = StandardScaler()
X_num = scaler.fit_transform(X_num)
y = df['target'].values

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

# Perform train-test split
X_text_temp, X_text_test, X_mask_temp, X_mask_test, X_num_temp, X_num_test, y_temp, y_test = train_test_split(
    X_text_np, X_mask_np, X_num, y, test_size=0.15, random_state=42
)

X_text_train, X_text_val, X_mask_train, X_mask_val, X_num_train, X_num_val, y_train, y_val = train_test_split(
    X_text_temp, X_mask_temp, X_num_temp, y_temp, test_size=0.1765, random_state=42
)

# Compute class weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

# Load pretrained BERT model
bert_model = TFBertModel.from_pretrained(bert_model_name)

# Build model
text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='text_input')
mask_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype=tf.int32, name='mask_input')
bert_output = bert_model(text_input, attention_mask=mask_input)[1]

num_input = Input(shape=(X_num_train.shape[1],), name='num_input')
num_layer = Dense(32, activation='relu')(num_input)
num_layer = Dropout(0.3)(num_layer)

combined = Concatenate()([bert_output, num_layer])
fc = Dense(64, activation='relu')(combined)
fc = Dropout(0.3)(fc)
output = Dense(len(label_encoder.classes_), activation='softmax')(fc)

model = Model(inputs=[text_input, mask_input, num_input], outputs=output)

# Compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train model
batch_size = 16
train_dataset = tf.data.Dataset.from_tensor_slices(({
    'text_input': X_text_train, 'mask_input': X_mask_train, 'num_input': X_num_train
}, y_train)).shuffle(1024).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices(({
    'text_input': X_text_val, 'mask_input': X_mask_val, 'num_input': X_num_val
}, y_val)).batch(batch_size)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
with tf.device('/GPU:0'):  # Use GPU 0 explicitly
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=25,
        callbacks=[early_stopping],
        class_weight=class_weights_dict,
        verbose=1
    )

# Evaluate and generate classification report
test_dataset = tf.data.Dataset.from_tensor_slices(({
    'text_input': X_text_test, 'mask_input': X_mask_test, 'num_input': X_num_test
}, y_test)).batch(batch_size)
with tf.device('/GPU:0'):  # Use GPU 0 explicitly
    y_pred = model.predict(test_dataset)
    y_pred_classes = tf.argmax(y_pred, axis=1).numpy()

from sklearn.metrics import classification_report
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
