import json
import os
from natsort import natsorted
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig

root = "C:\\Users\\Research chair\\Desktop\\Roken Alhewar Datasset\\latest_dataset(en)"

accepted_file = os.listdir(os.path.join(root, "Accepted_Islam_splits"))
wants_file = os.listdir(os.path.join(root, "Wants_to_Convert_splits"))
interested_file = os.listdir(os.path.join(root, "Interested_in_Islam_splits"))

accepted_file = natsorted(accepted_file)
wants_file = natsorted(wants_file)
interested_file = natsorted(interested_file)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the Sentiment Score model 1as
model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
config = RobertaConfig.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, config=config)
# Move the model to the GPU
model.to(device)

def get_sentiment_scores(text):
    # Tokenize the text
    inputs = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt").to(device)
    # Get the model's output
    outputs = model(**inputs)
    # Get the sentiment score (logits)
    logits = outputs.logits
    # Convert logits to probabilities
    probabilities = torch.softmax(logits, dim=1)
    # Normalize the score to be between -1 and 1
    positive_score = probabilities[0][2].item()  # Assuming index 2 is for positive sentiment
    negative_score = probabilities[0][0].item()  # Assuming index 0 is for negative sentiment
    normalized_score = round(positive_score - negative_score, 4)
    return normalized_score


def add_sentiment_scores(file_list, folder_name):
    for file_name in file_list:
        file_path = os.path.join(root, folder_name, file_name)
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        daei_scores = []
        visitor_scores = []
        all_scores = []

        for message in data['messages']:
            sentiment_score = get_sentiment_scores(message['message'])
            message['message_sentiment_score'] = sentiment_score
            all_scores.append(sentiment_score)
            if 'Daei' in message:
                daei_scores.append(sentiment_score)
            elif 'Visitor' in message:
                visitor_scores.append(sentiment_score)

        data['answers_average_score'] = round(sum(daei_scores) / len(daei_scores), 4) if daei_scores else 0
        data['questions_average_score'] = round(sum(visitor_scores) / len(visitor_scores), 4) if visitor_scores else 0
        data['conversation_average_score'] = round(sum(all_scores) / len(all_scores), 4) if all_scores else 0

        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


add_sentiment_scores(accepted_file, "Accepted_Islam_splits")
add_sentiment_scores(wants_file, "Wants_to_Convert_splits")
add_sentiment_scores(interested_file, "Interested_in_Islam_splits")
