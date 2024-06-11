import pandas as pd

# Load the dataset from a CSV file into a pandas DataFrame
df = pd.read_csv('train.csv')

# Define a function to merge the assistance and user columns into a conversation format
def merge_conversation(row):
    return f"User: {row['User']} Assistance: {row['Assistance']}"

# Apply the function to each row in the DataFrame
df['conversation'] = df.apply(merge_conversation, axis=1)

# Select the new conversation column
conversation_df = df[['conversation']]

# Save the new DataFrame with conversations to a new CSV file
conversation_df.to_csv('conversations.csv', index=False)
