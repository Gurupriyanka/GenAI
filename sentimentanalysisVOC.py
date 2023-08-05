import streamlit as st
import pandas as pd
import openai
import json
from io import BytesIO

# Read API key and parameters from file
with open('api_key.txt', 'r') as f:
    api_info = f.read().strip().split('\n')
    API_KEY = api_info[0].split('=')[1].strip()
    api_base = api_info[1].split('=')[1].strip()
    api_type = api_info[2].split('=')[1].strip()
    api_version = api_info[3].split('=')[1].strip()

# Set OpenAI API key and parameters
openai.api_key = API_KEY
openai.api_base = api_base
openai.api_type = api_type
openai.api_version = api_version

# Set deployment ID
# DEPLOYMENT_ID = 'gpt-35-turbo'
DEPLOYMENT_ID = 'text-davinci-003'


# Function to perform sentiment analysis using OpenAI API
def perform_sentiment_analysis(comment):
    prompt = f"Open AI: Sentiment Data: {comment}\nSentiment Category: Identify the sentiment of " \
             f"data: (Positive, Negative, Neutral)\nImprovements: (Key areas of improvement for " \
             f"negative or neutral sentiment, limit to two areas)\nDelight: (Key areas of delight " \
             f"for positive sentiment, limit to two areas)"
    # print("prompt =", prompt)

    response = openai.Completion.create(
        engine=DEPLOYMENT_ID,

        prompt=prompt,
        max_tokens=500,
        n=1,
        temperature=0.0,
        top_p=1.0
    )
    # print("response =", response)

    sentiment_data = response.choices[0].text
    sentiment_data = sentiment_data.lstrip()
    sentiment_result = sentiment_data.split('\n')
    # print("sentiment data =", sentiment_data)
    # print("sentiment result =", sentiment_result)

    sentiment_category = None
    improvements = None
    delight = None

    for line in sentiment_result:
        if line.startswith("Improvements:"):
            improvements = line.split(":")[1].strip()
            improvements = ', '.join(improvements.split(',')[:2]).strip()
        elif line.startswith("Delight:"):
            delight = line.split(":")[1].strip()
            delight = ', '.join(delight.split(',')[:2]).strip()

    if "Negative" in sentiment_data:
        sentiment_category = "Negative"
    elif "Positive" in sentiment_data:
        sentiment_category = "Positive"
    else:
        sentiment_category = "Neutral"

    return sentiment_category, improvements, delight


# Function to convert sentiment score to a scale of 1 to 10
def convert_sentiment_score(score):
    if score is not None:
        return score * 9 + 1  # Scale the sentiment score to a range of 1 to 10
    else:
        return None

# Streamlit app
def main():
    st.title('Sentiment Analysis')
    st.write('Upload a CSV file with Team Name, Feedback Id, and Comments:')

    file = st.file_uploader("Upload CSV", type="csv")

    if file is not None:
        df = pd.read_csv(file)
        df['Sentiment Category'] = ''
        df['Sentiment Score'] = ''
        df['Areas of Improvement'] = ''
        df['Areas of Delight'] = ''

        for index, row in df.iterrows():
            team_name = row['Team Name']
            feedback_id = row['Feedback Id']
            comment = row['Comments']
            sentiment_category, improvements, delight = perform_sentiment_analysis(comment)

            if sentiment_category == 'Negative' or sentiment_category == 'Neutral':
                df.at[index, 'Areas of Improvement'] = improvements if improvements else ''
            elif sentiment_category == 'Positive':
                df.at[index, 'Areas of Delight'] = delight if delight else ''

            df.at[index, 'Sentiment Category'] = sentiment_category
            # Update other columns accordingly

        # Display the DataFrame
        st.write(df)

        # Create download button for the CSV file
        output_df = df[['Team Name', 'Feedback Id', 'Comments','Sentiment Category', 'Sentiment Score', 'Areas of Improvement', 'Areas of Delight']]
        st.download_button("Download CSV", output_df.to_csv(index=False), file_name='classified_data.csv', mime='text/csv')


if __name__ == '__main__':
    main()
