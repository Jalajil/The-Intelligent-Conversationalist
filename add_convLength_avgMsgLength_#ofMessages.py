# Renamed this file from Add number_of_session, length feature.py to add_convLength_avgMsgLength_#ofMessages
import os
import json
from natsort import natsorted
import shutil

# Root directory containing the original folders with JSON data files
root = 'C:\\Users\\Research chair\\Desktop\\Roken Alhewar Datasset\\latest_dataset(en)\\Data that has 100%'
# Folders for each category of data
folders = ["Accepted_Islam_splits", "Wants_to_Convert_splits", "Interested_in_Islam_splits"]

# New root directory to store the modified JSON files
new_root = 'C:\\Users\\Research chair\\Desktop\\Roken Alhewar Datasset\\latest_dataset(en)\\Data that has 100% with new features'
os.makedirs(new_root, exist_ok=True)

# List to store paths of all JSON files
file_paths = []

# Collect file paths for each JSON file and create corresponding folders in new_root
for folder_name in folders:
    # Construct the path to each original folder
    folder_path = os.path.join(root, folder_name)
    # Create the corresponding folder in the new directory
    new_folder_path = os.path.join(new_root, folder_name)
    os.makedirs(new_folder_path, exist_ok=True)

    # List all files in the folder and sort them naturally
    files = os.listdir(folder_path)
    files = natsorted(files)

    # Loop through each file in the folder
    for file_name in files:
        # Construct the full path to the JSON file in the original and new folders
        file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(new_folder_path, file_name)

        # Add both file paths for processing
        file_paths.append((file_path, new_file_path))

# Adding conversation features and saving to new files in the new directory
for original_file_path, new_file_path in file_paths:
    # Open and read the JSON data
    with open(original_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

        # Initialize feature variables for conversation metrics
        total_words = 0
        num_messages = 0
        avg_message_length = 0

        # Check if there are messages in the data
        messages = data.get('messages', [])
        if messages:
            # Calculate the number of messages
            num_messages = len(messages)

            # Calculate total words and average message length
            for message in messages:
                message_text = message.get('message', "")  # Access 'message' field for the text
                word_count = len(message_text.split())
                total_words += word_count

            # Calculate average message length, avoiding division by zero
            avg_message_length = total_words / num_messages if num_messages else 0

        # Add the new features
        data['conversation_length'] = total_words  # Total words in the conversation
        data['average_message_length'] = avg_message_length  # Average words per message
        data['number_of_messages'] = num_messages  # Total message count in the conversation

        # Write the updated data back to the new JSON file in the new directory
        with open(new_file_path, 'w', encoding='utf-8') as fw:
            json.dump(data, fw, ensure_ascii=False, indent=2)

print("New JSON files with conversation features added in the new directory.")
