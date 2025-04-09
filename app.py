import streamlit as st
from transformers import TFAutoModelForSequenceClassification, AutoTokenizer
import tensorflow as tf
from PIL import Image

# Loading the model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained("c:/Data_Science/IH_Notebooks/tweet_emotion_recognition")
tokenizer = AutoTokenizer.from_pretrained("c:/Data_Science/IH_Notebooks/tweet_emotion_recognition")

# Defining the prediction function
def mooder(text):
    """
    This function will return the mood of the text
    """
    encoded_text = tokenizer(text, return_tensors="tf", max_length=512, truncation=True, padding="longest")
    

    output = model(**encoded_text)
    scores = output.logits[0].numpy()

    moods = {
        "anger": scores[0],
        "joy": scores[1],
        "fear": scores[2],
        "love": scores[3],
        "surprise": scores[4],
        "sadness": scores[5]
    }

    max_mood = max(moods, key=moods.get)

    if max_mood == 'joy':
        response = "I'm glad to hear you are having a great time. May you feel this way for a long time."
    elif max_mood == 'anger':
        response = 'I understand how you can feel frustrated and I wish you feel better soon.'
    elif max_mood == 'fear':
        response = 'I understand how you can be concerned. Please connect with a loved one to seek support.'
    elif max_mood == 'love':
        response = 'Feeling love is the best feeling one could experience. May this long last for you and your loved one.'
    elif max_mood == 'surprise':
        response = 'You seem surprised! I hope the surprise was a good one for you!'
    else:
        response = "I'm sorry to hear you are sad. If you need support, please reach out to one of your loved ones."

    return max_mood, response

# Streamlit interface
st.set_page_config(page_title="Emotion Detection App", layout="centered")

# Background image
img = Image.open('c:/Data_Science/IH_Notebooks/tweet_emotion_recognition/emotions.jpg')
st.image(img, use_column_width=True)

# Header text
st.title("Welcome to the Emotion Detection App!")
st.write("""
This app uses a **RoBERTa model** fine-tuned to detect emotions in text. 
Simply type a sentence or paragraph, and the app will predict the emotion conveyed in the text. 
The emotions include: **Sadness, Joy, Love, Anger, Fear, and Surprise.**
""")

# User input
user_input = st.text_area("Enter a message:", height=200)

# Prediction on button click
if st.button("Predict Emotion"):
    if user_input:
        max_mood, response = mooder(user_input)
        st.write(f"The predicted emotion is: **{max_mood}**")
        st.write(response)
    else:
        st.write("Please enter some text to predict the emotion.")
