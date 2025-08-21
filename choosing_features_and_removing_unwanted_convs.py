import json
import os

# Initialize a counter for naming output files
count = 1

# Get a list of all files in the specified directory
listOfFiles = os.listdir("C:\\Users\\Research chair\\Desktop\\Roken Alhewar Datasset\\IslamChatScripts_Wants")
print(listOfFiles)
print(len(listOfFiles))

# Process each file in the list
while listOfFiles:
    # Open and read the content of the first file in the list
    with open(file=f"C:\\Users\\Research chair\\Desktop\\Roken Alhewar Datasset\\IslamChatScripts_Wants\\{listOfFiles.pop(0)}", mode="r") as file:
        data = file.read()

    # Parse the JSON data from the file
    js = json.loads(data)

    # Initialize lists to store messages and the final structured data
    msg = []
    msgs = []

    # Iterate over each entry in the JSON data
    for i in js:
        # Skip this entry if visitor's name is "Visitor"
        if i["visitor_name"] == "Visitor":
            continue

        if len(i['messages']) < 5:
            continue  # Skip this entry if it has less than 5 messages

        rate = "not rated"  # Default rating if not found
        visitor = i["visitor_name"]  # Get the visitor's name
        General_Id = i["id"]
        duration = i["duration"]
        chat_id = i["lc3"]["chat_id"]
        city = i["visitor"]["city"]
        country = i["visitor"]["country"]
        region = i["visitor"]["region"]
        timezone = i["visitor"]["timezone"]
        visitor_Id = i["visitor"]["id"]
        r = i["events"]  # Get the events related to the chat
        i = i['messages']  # Get the messages in the chat
        msg = []

        # Process each message in the chat
        for j in i:
            if 'visitor' == j['user_type']:
                jsdata = {"Visitor": j['author_name'], "message": j['text'], "date": j['date']}
            else:
                jsdata = {"Daei": j['author_name'], "message": j['text'], "date": j['date']}
            msg.append(jsdata)

        # Check for rating information in the events
        for k in r:
            if 'text_vars' in k.keys() and 'score' in k['text_vars'].keys():
                rate = k['text_vars']['score']

        # Create a dictionary with messages and rating
        dict = {"General_ID": General_Id, "chat_id": chat_id, "Visitor": {"country": country, "region": region, "city": city, "timezone": timezone, "ID": visitor_Id},
                "messages": msg, "duration": duration, "rate": rate}
        msgs.append(dict)
        print(dict)

    # Write the structured data to a new JSON file
    for i in msgs:
        with open(f"C:\\Users\\Research chair\\Desktop\\Roken Alhewar Datasset\\latest_dataset\\Wants_to_Convert\\Wants_to_Convert{count}.json", "w") as f:
            json.dump(i, f)
            count += 1
