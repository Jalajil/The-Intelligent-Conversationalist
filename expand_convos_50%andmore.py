import json
import os

# Define the source and destination directories
source_dir = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset\latest_dataset(en)\Data that has 100% with new features"
dest_dir = r"C:\Users\Research chair\Desktop\Roken Alhewar Datasset\latest_dataset(en)\Data with more than 50%"

# Ensure the destination directory exists
os.makedirs(dest_dir, exist_ok=True)

# Define percentages for splitting
percentages = [50, 60, 70, 80, 90, 100]

# Function to calculate updated values for the split
def update_values(data, messages):
    daei_messages = [msg for msg in messages if "Daei" in msg]
    visitor_messages = [msg for msg in messages if "Visitor" in msg]

    data["answers_average_score"] = round(
        sum(msg["message_sentiment_score"] for msg in daei_messages) / len(daei_messages)
        if daei_messages else 0, 4
    )
    data["questions_average_score"] = round(
        sum(msg["message_sentiment_score"] for msg in visitor_messages) / len(visitor_messages)
        if visitor_messages else 0, 4
    )
    data["conversation_average_score"] = round(
        sum(msg["message_sentiment_score"] for msg in messages) / len(messages)
        if messages else 0, 4
    )
    data["conversation_length"] = sum(len(msg["message"].split()) for msg in messages)
    data["number_of_messages"] = len(messages)
    data["average_message_length"] = round(
        data["conversation_length"] / data["number_of_messages"]
        if data["number_of_messages"] > 0 else 0, 4
    )

    return data


# Iterate through each folder in the source directory
for folder_name in os.listdir(source_dir):
    folder_path = os.path.join(source_dir, folder_name)
    if os.path.isdir(folder_path):
        new_folder_path = os.path.join(dest_dir, folder_name)
        os.makedirs(new_folder_path, exist_ok=True)

        # Iterate through each file in the folder
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                # Read the JSON content
                with open(file_path, 'r', encoding='utf-8') as file:
                    try:
                        data = json.load(file)
                    except json.JSONDecodeError:
                        print(f"Skipping invalid JSON file: {file_name}")
                        continue

                # Ensure 'messages' key exists for splitting
                if "messages" in data and isinstance(data["messages"], list):
                    messages = data["messages"]

                    # Split the 'messages' array by percentages
                    for percentage in percentages:
                        split_point = int(len(messages) * (percentage / 100))
                        split_messages = messages[:split_point]

                        # Update the data for the split
                        split_data = data.copy()
                        split_data["messages"] = split_messages
                        split_data = update_values(split_data, split_messages)

                        # Create the new file name and path
                        base_name, ext = os.path.splitext(file_name)
                        new_file_name = f"{base_name}_{percentage}%{ext}"
                        new_file_path = os.path.join(new_folder_path, new_file_name)

                        # Save the split JSON data
                        with open(new_file_path, 'w', encoding='utf-8') as new_file:
                            json.dump(split_data, new_file, ensure_ascii=False, indent=4)

print("Splitting completed. The new JSON files are saved in the destination directory.")