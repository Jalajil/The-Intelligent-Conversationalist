import json
import os
from natsort import natsorted
import shutil
import random

def contains_both_roles(messages):
    daei_exists = any("Daei" in msg for msg in messages)
    visitor_exists = any("Visitor" in msg for msg in messages)
    return daei_exists and visitor_exists

def adjust_first_split(messages_dict, total_messages, min_percentage=10, min_messages=5):
    split_index = 0
    first_split = []
    while (not contains_both_roles(first_split) or len(first_split) < min_messages) and split_index < total_messages:
        first_split.append(messages_dict[split_index])
        split_index += 1
    first_split_percentage = (split_index / total_messages) * 100
    while first_split_percentage < min_percentage and split_index < total_messages:
        first_split.append(messages_dict[split_index])
        split_index += 1
        first_split_percentage = (split_index / total_messages) * 100
    return first_split, split_index, first_split_percentage

def split_file(file_path, percentages):
    """ Split the messages in a JSON file into multiple parts based on given percentages. """
    with open(file_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    messages_dict = original_data['messages']
    splits = {}
    total_messages = len(messages_dict)

    # Adjust the first split to ensure both 'Daei' and 'Visitor' are present
    first_split, split_index, first_split_percentage = adjust_first_split(messages_dict, total_messages)
    splits[int(first_split_percentage)] = {
        'General_ID': original_data.get('General_ID', None),
        'chat_id': original_data.get('chat_id', None),
        'Visitor': original_data.get('Visitor', None),
        'messages': first_split,
        'duration': original_data.get('duration', None),
        'rate': original_data.get('rate', None)  # Keep the 'rate' key if it exists
    }

    # Calculate the remaining splits based on the percentages
    for percentage in percentages:
        if percentage <= first_split_percentage:
            continue  # Skip any percentage smaller than the first split

        split_index_for_percentage = int(total_messages * (percentage / 100))
        split_messages = messages_dict[:split_index_for_percentage]
        splits[percentage] = {
            'General_ID': original_data.get('General_ID', None),
            'chat_id': original_data.get('chat_id', None),
            'Visitor': original_data.get('Visitor', None),
            'messages': split_messages,
            'duration': original_data.get('duration', None),
            'rate': original_data.get('rate', None)
        }

    return splits

def save_splits(splits, category, file_name, root):
    base_name = os.path.splitext(file_name)[0]
    category_folder = os.path.join(root, f"{category}_splits")
    os.makedirs(category_folder, exist_ok=True)
    for percentage, data in splits.items():
        output_file = os.path.join(category_folder, f"{base_name}_{percentage}%.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

def rename_files(category_path):
    files = natsorted(os.listdir(category_path))
    for index, file_name in enumerate(files, start=1):
        new_name = f"{category_path.split(os.sep)[-1]}{index}.json"
        os.rename(os.path.join(category_path, file_name), os.path.join(category_path, new_name))

def check_files_for_messages(root, categories):
    files_with_less_than_5_messages = []

    for category in categories:
        category_path = os.path.join(root, category)
        files = os.listdir(category_path)
        files = natsorted(files)

        for file_name in files:
            file_path = os.path.join(category_path, file_name)
            if os.path.getsize(file_path) == 0:
                continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if len(data['messages']) < 5:
                        files_with_less_than_5_messages.append(file_name)
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {file_name}")

    if files_with_less_than_5_messages:
        print("Files with less than 5 messages:")
        for file_name in files_with_less_than_5_messages:
            print(file_name)
    else:
        print("There are no files with less than 5 messages.")


def create_training_validation(base_dir, categories, train_ratio=0.7):
    for category in categories:
        category_path = os.path.join(base_dir, category)
        files = os.listdir(category_path)

        # Group files by rank number
        rank_groups = {}
        for file in files:
            parts = file.split('_')
            if len(parts) == 3:
                rank = parts[1]  # Assuming file format is category_rank_percentage
            elif len(parts) == 4:
                rank = parts[2]  # Assuming file format is category_rank_percentage
            else:
                continue  # Skip files that don't match expected format

            if rank not in rank_groups:
                rank_groups[rank] = []
            rank_groups[rank].append(os.path.join(category_path, file))

        # Split rank groups into training and validation sets
        ranks = list(rank_groups.keys())
        random.shuffle(ranks)
        split_index = int(len(ranks) * train_ratio)
        train_ranks = ranks[:split_index]
        val_ranks = ranks[split_index:]

        # Create directories for training and validation sets
        train_dir = os.path.join(base_dir, 'train', category)
        val_dir = os.path.join(base_dir, 'validation', category)
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Copy files to respective directories
        for rank in train_ranks:
            for file in rank_groups[rank]:
                shutil.copy(file, os.path.join(train_dir, os.path.basename(file)))

        for rank in val_ranks:
            for file in rank_groups[rank]:
                shutil.copy(file, os.path.join(val_dir, os.path.basename(file)))


root = 'C:\\Users\\Research chair\\Desktop\\Roken Alhewar Datasset\\latest_dataset(en)'
categories = ["Accepted_Islam_splits", "Wants_to_Convert_splits", "Interested_in_Islam_splits"]
percentages = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

# for category in categories:
#     category_path = os.path.join(root, category)
#     files = os.listdir(category_path)
#     files = natsorted(files)
#     for file_name in files:
#         file_path = os.path.join(category_path, file_name)
#         if os.path.getsize(file_path) == 0:
#             print(f"Skipping empty file: {file_name}")
#             continue
#         try:
#             with open(file_path, 'r', encoding='utf-8') as f:
#                 data = json.load(f)
#                 if len(data['messages']) < 5:
#                     print(f"Deleting file with less than 5 messages: {file_name}")
#                     f.close()  # Ensure the file is closed before deleting
#                     os.remove(file_path)
#                 else:
#                     splits = split_file(file_path, percentages)
#                     save_splits(splits, category, file_name, root)
#         except json.JSONDecodeError:
#             print(f"Error decoding JSON in file: {file_name}")
#     rename_files(category_path)
#
#
# check_files_for_messages(root, categories)

# Creating the training and validation sets
create_training_validation(root, categories)