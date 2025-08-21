import os
import re
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.metrics import classification_report
from datetime import datetime
from natsort import natsorted
import pickle  # Import pickle to save the tokenizer and scaler

# Download stopwords and WordNet data
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Adjust the GPU index as per your system configuration
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU is available and configured.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU is not available.")

# Root path for the dataset
root_data = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset\latest_dataset(en)\more than 50% without Wants_to_Convert\Data with more than 50%"

# Initialize WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    stop_words = set(stopwords.words('english'))
    domain_stopwords = {'islam', 'convert', 'religion'}
    stop_words.update(domain_stopwords)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
    return text

# Load data function with duration calculation
def load_data(root_path):
    texts, labels, scores, number_of_sessions, conversation_lengths, average_message_lengths, number_of_messages, durations = [], [], [], [], [], [], [], []
    for category_folder in os.listdir(root_path):
        category_path = os.path.join(root_path, category_folder)
        label = 0 if 'Interested_in_Islam' in category_folder else 1
        files = natsorted(os.listdir(category_path))
        for filename in files:
            file_path = os.path.join(category_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # Extract messages and timestamps
                messages = data['messages']
                conversation = " ".join([f"{list(msg.keys())[0]}: {msg['message']}" for msg in messages])
                conversation = preprocess_text(conversation)
                texts.append(conversation)
                labels.append(label)
                scores.append(data.get('conversation_average_score', 0.0))
                number_of_sessions.append(data.get('number_of_session', 0))
                conversation_lengths.append(data.get('conversation_length', 0))
                average_message_lengths.append(data.get('average_message_length', 0.0))
                number_of_messages.append(data.get('number_of_messages', 0))

                # Calculate duration from the date field
                date_format = "%a, %m/%d/%y %I:%M:%S %p"
                timestamps = [
                    datetime.strptime(msg['date'], date_format) for msg in messages if 'date' in msg
                ]
                if timestamps:
                    first_timestamp = min(timestamps)
                    last_timestamp = max(timestamps)
                    duration = (last_timestamp - first_timestamp).total_seconds()
                    durations.append(duration)
                else:
                    durations.append(0)  # Default to 0 if no dates are present
    return texts, labels, scores, number_of_sessions, conversation_lengths, average_message_lengths, number_of_messages, durations

# Load data
texts, labels, scores, number_of_sessions, conversation_lengths, average_message_lengths, number_of_messages, durations = load_data(root_data)

# Split data into train, validation, and test sets
train_texts, temp_texts, train_labels, temp_labels, train_scores, temp_scores, train_sessions, temp_sessions, \
train_conv_lengths, temp_conv_lengths, train_avg_msg_lengths, temp_avg_msg_lengths, train_num_msgs, temp_num_msgs, \
train_durations, temp_durations = train_test_split(
    texts, labels, scores, number_of_sessions, conversation_lengths, average_message_lengths, number_of_messages, durations,
    test_size=0.3, random_state=42, stratify=labels)

val_texts, test_texts, val_labels, test_labels, val_scores, test_scores, val_sessions, test_sessions, \
val_conv_lengths, test_conv_lengths, val_avg_msg_lengths, test_avg_msg_lengths, val_num_msgs, test_num_msgs, \
val_durations, test_durations = train_test_split(
    temp_texts, temp_labels, temp_scores, temp_sessions, temp_conv_lengths, temp_avg_msg_lengths, temp_num_msgs, temp_durations,
    test_size=0.5, random_state=42, stratify=temp_labels)

# One-hot encode the labels
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels, num_classes=2)
val_labels = to_categorical(val_labels, num_classes=2)
test_labels = to_categorical(test_labels, num_classes=2)

# Tokenization and padding
vocab_size = 100000
max_length = 2500

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)

# Save the tokenizer
tokenizer_path = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset\tokenizer.pickle"
with open(tokenizer_path, 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Tokenizer saved to {tokenizer_path}")

train_sequences = tokenizer.texts_to_sequences(train_texts)
val_sequences = tokenizer.texts_to_sequences(val_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

train_padded = pad_sequences(train_sequences, maxlen=max_length, padding='post', truncating='post')
val_padded = pad_sequences(val_sequences, maxlen=max_length, padding='post', truncating='post')
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding='post', truncating='post')

# Load GloVe embeddings
def load_glove_embeddings(glove_file_path, tokenizer, embedding_dim=300):
    embeddings_index = {}
    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            except ValueError:
                pass
    embedding_matrix = np.zeros((vocab_size, embedding_dim))
    for word, i in tokenizer.word_index.items():
        if i < vocab_size:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

embedding_dim = 300
glove_file_path = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset\latest_dataset(en)\glove.42B.300d.txt"
embedding_matrix = load_glove_embeddings(glove_file_path, tokenizer, embedding_dim)

# Normalize features, including duration
train_features = np.array(list(zip(
    train_scores, train_sessions, train_conv_lengths, train_avg_msg_lengths, train_num_msgs, train_durations)))
val_features = np.array(list(zip(
    val_scores, val_sessions, val_conv_lengths, val_avg_msg_lengths, val_num_msgs, val_durations)))
test_features = np.array(list(zip(
    test_scores, test_sessions, test_conv_lengths, test_avg_msg_lengths, test_num_msgs, test_durations)))

scaler = MinMaxScaler()
train_features = scaler.fit_transform(train_features)
val_features = scaler.transform(val_features)
test_features = scaler.transform(test_features)

# Save the scaler
scaler_path = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset\scaler.pickle"
with open(scaler_path, 'wb') as handle:
    pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
print(f"Scaler saved to {scaler_path}")

# Dataset preparation
batch_size = 128
train_data = tf.data.Dataset.from_tensor_slices(
    ({'text': train_padded, 'features': train_features}, train_labels)).shuffle(len(train_texts)).batch(batch_size)
val_data = tf.data.Dataset.from_tensor_slices(
    ({'text': val_padded, 'features': val_features}, val_labels)).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices(
    ({'text': test_padded, 'features': test_features}, test_labels)).batch(batch_size)

# Calculate class weights
class_counts = np.sum(train_labels, axis=0)
class_weights = {i: len(train_labels) / (len(class_counts) * count) for i, count in enumerate(class_counts)}
weight_multiplier = 1.1  # Adjust this value if needed
class_weights[1] *= weight_multiplier
print("Class weights:", class_weights)

# Model definition
def build_improved_model():
    # Text input
    text_input = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32, name='text')
    features_input = tf.keras.layers.Input(shape=(train_features.shape[1],), dtype=tf.float32, name='features')

    # Embedding layer with trainable pre-trained GloVe embeddings
    embedding = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                          weights=[embedding_matrix], input_length=max_length, trainable=True)(text_input)
    embedding = tf.keras.layers.SpatialDropout1D(0.3)(embedding)

    # Conv1D for feature extraction
    conv = tf.keras.layers.Conv1D(128, kernel_size=5, activation='relu', padding='same')(embedding)
    conv = tf.keras.layers.MaxPooling1D(pool_size=2)(conv)

    # Bi-LSTM with Layer Normalization
    lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True, dropout=0.3))(conv)
    lstm = tf.keras.layers.LayerNormalization()(lstm)

    # Multi-head Attention with pooling
    attention = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=128)(lstm, lstm)
    avg_pool = tf.keras.layers.GlobalAveragePooling1D()(attention)
    max_pool = tf.keras.layers.GlobalMaxPooling1D()(attention)
    attention_output = tf.keras.layers.Concatenate()([avg_pool, max_pool])

    # Dense layer for features
    dense_features = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(features_input)

    # Combine text and feature inputs
    combined = tf.keras.layers.Concatenate()([attention_output, dense_features])
    dense = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.L2(0.01))(combined)
    dense = tf.keras.layers.Dropout(0.5)(dense)

    # Output layer
    classifier = tf.keras.layers.Dense(2, activation='softmax')(dense)

    # Model definition
    model = tf.keras.Model(inputs=[text_input, features_input], outputs=classifier)
    return model

# Build and compile the improved model
improved_model = build_improved_model()

# Compile the model
improved_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                       loss='categorical_crossentropy',
                       metrics=['accuracy'])

# Define callbacks
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6)
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = improved_model.fit(
    train_data,
    validation_data=val_data,
    epochs=50,
    class_weight=class_weights,  # Alternatively, use balanced class sampling
    callbacks=[early_stopping, reduce_lr]
)

# Evaluate the model
test_loss, test_accuracy = improved_model.evaluate(test_data)
test_predictions = improved_model.predict(test_data)
test_predictions_classes = np.argmax(test_predictions, axis=1)

# Print results
print(f"Test Accuracy: {test_accuracy}")
print(classification_report(np.argmax(test_labels, axis=1), test_predictions_classes, target_names=['Interested_in_Islam', 'Accepted_Islam']))

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.plot(history.history['loss'], label='Train Loss')
ax1.plot(history.history['val_loss'], label='Validation Loss')
ax1.set_title('Training and Validation Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()

ax2.plot(history.history['accuracy'], label='Train Accuracy')
ax2.plot(history.history['val_accuracy'], label='Validation Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()

plt.tight_layout()
plt.show()

# Save the model in the TensorFlow SavedModel format
model_save_path = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset"
improved_model.save(model_save_path)
print(f"Model saved to {model_save_path}")
