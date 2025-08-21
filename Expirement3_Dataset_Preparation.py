import json
import os
from tqdm import tqdm

# Additional imports for model prediction
import tensorflow as tf
import numpy as np
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import pickle

# Suppress TensorFlow warnings and info messages (optional)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Download NLTK resources if not already downloaded
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Set memory growth to prevent TensorFlow from allocating all GPU memory upfront
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        # Optionally, set which GPU(s) to use (if you have multiple GPUs)
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        print("GPU is available and configured.")
    except RuntimeError as e:
        print(f"Error configuring GPU: {e}")
else:
    print("GPU is not available.")

def create_filtered_dataset(data_path, output_dir, min_prev_questions=5, min_follow_questions=1,
                            max_follow_questions=5):
    """
    Process JSON files to filter and transform chat data based on specified conditions.

    Args:
        data_path (str): Path to the main directory containing the three folders.
        output_dir (str): Directory to save the new JSON files.
        min_prev_questions (int): Minimum number of preceding questions for an answer to qualify.
        min_follow_questions (int): Minimum number of following questions for an answer to qualify.
        max_follow_questions (int): Maximum number of questions to include in the average calculation.

    Returns:
        None
    """
    # Load the model
    model_save_path = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset"
    loaded_model = tf.keras.models.load_model(model_save_path)

    # Load the tokenizer
    tokenizer_path = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset\tokenizer.pickle"
    with open(tokenizer_path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    print(f"Tokenizer loaded from {tokenizer_path}")

    # Load the scaler
    scaler_path = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset\scaler.pickle"
    with open(scaler_path, 'rb') as handle:
        scaler = pickle.load(handle)
    print(f"Scaler loaded from {scaler_path}")

    # Preprocess text function
    lemmatizer = WordNetLemmatizer()
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        stop_words = set(stopwords.words('english'))
        domain_stopwords = {'islam', 'convert', 'religion'}
        stop_words.update(domain_stopwords)
        text = ' '.join([word for word in text.split() if word not in stop_words])
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split()])
        return text

    # Parameters for tokenization and padding
    max_length = 2500  # Same as used during model training

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Map input subfolders to output folders
    subfolder_to_output_folder = {
        "Accepted_Islam_splits": "Accepted_Islam",
        "Wants_to_Convert_splits": "Wants_to_Convert",
        "Interested_in_Islam_splits": "Interested_in_Islam"
    }

    # Iterate through each folder in the main directory
    subfolders = ["Accepted_Islam_splits", "Wants_to_Convert_splits", "Interested_in_Islam_splits"]
    for subfolder in subfolders:
        folder_path = os.path.join(data_path, subfolder)
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.json')]

        for file in tqdm(files, desc=f"Processing {subfolder}"):
            with open(file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Extract messages
            messages = data.get("messages", [])
            if not messages:
                continue

            # Combine consecutive answers
            combined_messages = []
            current_combined_answer = {"message": "", "sentiment_scores": [], "dates": [], "sender": "Daei"}

            for msg in messages:
                if "Daei" in msg:  # If it's an answer
                    if current_combined_answer["message"]:
                        current_combined_answer["message"] += " " + msg["message"]
                    else:
                        current_combined_answer["message"] = msg["message"]
                    current_combined_answer["sentiment_scores"].append(msg.get("message_sentiment_score", 0))
                    if 'date' in msg:
                        current_combined_answer["dates"].append(msg['date'])
                else:  # If it's a question
                    if current_combined_answer["message"]:
                        # Finalize the current combined answer
                        avg_sentiment = sum(current_combined_answer["sentiment_scores"]) / len(
                            current_combined_answer["sentiment_scores"])
                        combined_messages.append({
                            "sender": "Daei",
                            "message": current_combined_answer["message"],
                            "message_sentiment_score": round(avg_sentiment, 4),
                            "dates": current_combined_answer["dates"]
                        })
                        current_combined_answer = {"message": "", "sentiment_scores": [], "dates": [], "sender": "Daei"}
                    combined_messages.append({"sender": "Visitor", **msg})

            # Add the last combined answer if any
            if current_combined_answer["message"]:
                avg_sentiment = sum(current_combined_answer["sentiment_scores"]) / len(
                    current_combined_answer["sentiment_scores"])
                combined_messages.append({
                    "sender": "Daei",
                    "message": current_combined_answer["message"],
                    "message_sentiment_score": round(avg_sentiment, 4),
                    "dates": current_combined_answer["dates"]
                })

            # Filter answers based on conditions
            valid_answers_found = False
            daei_order = 1  # Initialize the order for valid Daei messages
            base_file_name = os.path.splitext(os.path.basename(file))[0]  # Extract the base name of the file
            for i, msg in enumerate(combined_messages):
                if msg.get("sender") == "Daei":
                    # Collect preceding Visitor messages
                    prev_questions = []
                    for m in reversed(combined_messages[:i]):
                        if m.get("sender") == "Visitor":
                            prev_questions.append(m)
                            if len(prev_questions) == min_prev_questions:  # Stop at min_prev_questions
                                break
                    prev_questions.reverse()  # Restore chronological order

                    # Skip if fewer than min_prev_questions
                    if len(prev_questions) < min_prev_questions:
                        continue

                    # Calculate average sentiment score for preceding questions
                    avg_preceding_sentiment = round(
                        sum(m.get("message_sentiment_score", 0) for m in prev_questions) / len(prev_questions), 4
                    )

                    # Collect following Visitor messages (limit to max_follow_questions for averaging)
                    follow_questions = [m for m in combined_messages[i + 1:] if m.get("sender") == "Visitor"][:max_follow_questions]

                    # Calculate following average score (if applicable)
                    following_average_score = round(
                        sum(m.get("message_sentiment_score", 0) for m in follow_questions) / len(follow_questions),
                        4
                    ) if follow_questions else 0

                    # Check if the answer meets the conditions
                    if len(follow_questions) >= min_follow_questions:
                        valid_answers_found = True

                        # Assign the order to the valid Daei message
                        msg["order"] = daei_order

                        # Create a new conversation
                        new_data = {
                            "General_ID": data.get("General_ID"),
                            "Chat_ID": data.get("chat_id"),
                            "Daei_message": msg["message"],
                            "Daei_sentiment_score": round(msg["message_sentiment_score"], 4),
                            "order": daei_order,
                            "Preceding_average_score": avg_preceding_sentiment,
                            "Following_average_score": following_average_score,
                            "Number_of_messages": len(prev_questions) + len(follow_questions) + 1,
                            "Answers_average_score": round(msg["message_sentiment_score"], 4),
                            "Preceding_questions": [m["message"] for m in prev_questions],
                            "Following_questions": [m["message"] for m in follow_questions]
                        }

                        # Prepare for Label_before
                        # First preceding question
                        first_preceding_question = prev_questions[-1]

                        # Messages up to and including the first preceding question
                        idx_first_preceding = combined_messages.index(first_preceding_question)
                        conversation_before = combined_messages[:idx_first_preceding + 1]

                        # Construct conversation text with commas
                        conversation_text_before = ", ".join(
                            [f"{m['sender']}: {m['message']}" for m in conversation_before]
                        )

                        # Preprocess text for prediction
                        preprocessed_text_before = preprocess_text(" ".join([m['message'] for m in conversation_before]))

                        # Tokenize and pad
                        sequence_before = tokenizer.texts_to_sequences([preprocessed_text_before])
                        padded_sequence_before = pad_sequences(sequence_before, maxlen=max_length, padding='post', truncating='post')

                        # Prepare features
                        message_scores_before = [m.get('message_sentiment_score', 0.0) for m in conversation_before]
                        conversation_average_score_before = sum(message_scores_before) / len(message_scores_before) if message_scores_before else 0.0

                        number_of_session_before = data.get('number_of_session', 0)
                        conversation_length_before = len(conversation_before)
                        message_lengths_before = [len(m['message'].split()) for m in conversation_before if 'message' in m]
                        average_message_length_before = sum(message_lengths_before) / len(message_lengths_before) if message_lengths_before else 0.0
                        number_of_messages_before = len(conversation_before)
                        # Duration
                        date_format = "%a, %m/%d/%y %I:%M:%S %p"
                        timestamps_before = []
                        for m in conversation_before:
                            date_str = m.get('date') or (m.get('dates')[0] if 'dates' in m and m['dates'] else None)
                            if date_str:
                                try:
                                    timestamps_before.append(datetime.strptime(date_str, date_format))
                                except ValueError:
                                    pass
                        if timestamps_before:
                            first_timestamp_before = min(timestamps_before)
                            last_timestamp_before = max(timestamps_before)
                            duration_before = (last_timestamp_before - first_timestamp_before).total_seconds()
                        else:
                            duration_before = 0.0

                        # Prepare features array
                        features_before = np.array([[
                            conversation_average_score_before,
                            number_of_session_before,
                            conversation_length_before,
                            average_message_length_before,
                            number_of_messages_before,
                            duration_before
                        ]])

                        # Scale features
                        scaled_features_before = scaler.transform(features_before)

                        # Prepare input for the model
                        input_data_before = {'text': padded_sequence_before, 'features': scaled_features_before}

                        # Get prediction
                        prediction_before = loaded_model.predict(input_data_before)
                        probabilities_before = prediction_before[0]

                        # Extract probabilities for both classes
                        probability_interested_before = round(float(probabilities_before[0]), 4)  # Probability of 'Interested_in_Islam'
                        probability_accepted_before = round(float(probabilities_before[1]), 4)    # Probability of 'Accepted_Islam'

                        # Get predicted class
                        predicted_class_before = np.argmax(probabilities_before)
                        label_before = 'Accepted_Islam' if predicted_class_before == 1 else 'Interested_in_Islam'

                        # Assign probability corresponding to predicted class
                        if predicted_class_before == 1:
                            probability_before = probability_accepted_before
                        else:
                            probability_before = probability_interested_before

                        # Store in new_data
                        new_data['Label_before'] = label_before
                        new_data['Probability_before'] = probability_before
                        new_data['Conversation_before'] = conversation_text_before

                        # Prepare for Label_after
                        # Last following question
                        last_following_question = follow_questions[-1]

                        # Messages up to and including the last following question
                        idx_last_following = combined_messages.index(last_following_question)
                        conversation_after = combined_messages[:idx_last_following + 1]

                        # Construct conversation text with commas
                        conversation_text_after = ", ".join(
                            [f"{m['sender']}: {m['message']}" for m in conversation_after]
                        )

                        # Preprocess text for prediction
                        preprocessed_text_after = preprocess_text(" ".join([m['message'] for m in conversation_after]))

                        # Tokenize and pad
                        sequence_after = tokenizer.texts_to_sequences([preprocessed_text_after])
                        padded_sequence_after = pad_sequences(sequence_after, maxlen=max_length, padding='post', truncating='post')

                        # Prepare features
                        message_scores_after = [m.get('message_sentiment_score', 0.0) for m in conversation_after]
                        conversation_average_score_after = sum(message_scores_after) / len(message_scores_after) if message_scores_after else 0.0

                        number_of_session_after = data.get('number_of_session', 0)
                        conversation_length_after = len(conversation_after)
                        message_lengths_after = [len(m['message'].split()) for m in conversation_after if 'message' in m]
                        average_message_length_after = sum(message_lengths_after) / len(message_lengths_after) if message_lengths_after else 0.0
                        number_of_messages_after = len(conversation_after)
                        # Duration
                        timestamps_after = []
                        for m in conversation_after:
                            date_str = m.get('date') or (m.get('dates')[0] if 'dates' in m and m['dates'] else None)
                            if date_str:
                                try:
                                    timestamps_after.append(datetime.strptime(date_str, date_format))
                                except ValueError:
                                    pass
                        if timestamps_after:
                            first_timestamp_after = min(timestamps_after)
                            last_timestamp_after = max(timestamps_after)
                            duration_after = (last_timestamp_after - first_timestamp_after).total_seconds()
                        else:
                            duration_after = 0.0

                        # Prepare features array
                        features_after = np.array([[
                            conversation_average_score_after,
                            number_of_session_after,
                            conversation_length_after,
                            average_message_length_after,
                            number_of_messages_after,
                            duration_after
                        ]])

                        # Scale features
                        scaled_features_after = scaler.transform(features_after)

                        # Prepare input for the model
                        input_data_after = {'text': padded_sequence_after, 'features': scaled_features_after}

                        # Get prediction
                        prediction_after = loaded_model.predict(input_data_after)
                        probabilities_after = prediction_after[0]

                        # Extract probabilities for both classes
                        probability_interested_after = round(float(probabilities_after[0]), 4)  # Probability of 'Interested_in_Islam'
                        probability_accepted_after = round(float(probabilities_after[1]), 4)    # Probability of 'Accepted_Islam'

                        # Get predicted class
                        predicted_class_after = np.argmax(probabilities_after)
                        label_after = 'Accepted_Islam' if predicted_class_after == 1 else 'Interested_in_Islam'

                        # Assign probability corresponding to predicted class
                        if predicted_class_after == 1:
                            probability_after = probability_accepted_after
                        else:
                            probability_after = probability_interested_after

                        # Store in new_data
                        new_data['Label_after'] = label_after
                        new_data['Probability_after'] = probability_after
                        new_data['Conversation_after'] = conversation_text_after

                        # Calculate target based on the specified conditions
                        prob_diff = round(probability_after - probability_before, 4)
                        if label_before == 'Interested_in_Islam' and label_after == 'Accepted_Islam':
                            new_data['target'] = 'Positive'
                        elif label_before == 'Accepted_Islam' and label_after == 'Interested_in_Islam':
                            new_data['target'] = 'Negative'
                        elif label_before == label_after:
                            if prob_diff >= 0.01:
                                new_data['target'] = 'Positive'
                            elif prob_diff <= -0.01:
                                new_data['target'] = 'Negative'
                            else:
                                new_data['target'] = 'Neutral'
                        else:
                            new_data['target'] = 'Neutral'

                        # Determine the output folder based on the original label
                        target_folder = subfolder_to_output_folder.get(subfolder, 'Unknown')

                        # Create the target folder if it doesn't exist
                        target_folder_path = os.path.join(output_dir, target_folder)
                        if not os.path.exists(target_folder_path):
                            os.makedirs(target_folder_path)

                        # Save to the target folder
                        new_file_name = f"{base_file_name}_{daei_order}.json"
                        output_file = os.path.join(target_folder_path, new_file_name)
                        with open(output_file, 'w', encoding='utf-8') as out_f:
                            json.dump(new_data, out_f, indent=4, ensure_ascii=False)

                        # Increment the order
                        daei_order += 1

            if not valid_answers_found:
                print(f"No valid answers in file: {file}. Skipping this file.")

    print(f"All files processed. Results saved in {output_dir}.")

# Usage
data_path = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset\latest_dataset(en)\test"
output_dir = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset\filtered_results"
create_filtered_dataset(data_path, output_dir)
