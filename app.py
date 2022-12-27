import streamlit as st
import soundfile as sf
import assemblyai
import transformers

# Set up a cache for the app
@st.cache
def get_recorder():
    # Create a recorder object
    recorder = sf.SoundRecorder(channels=1, samplerate=44100)
    return recorder

recorder = get_recorder()

# Set up the client with your API key
client = assemblyai.Client(api_key="YOUR_API_KEY_HERE")

# Load a BERT model for sentiment analysis
model = transformers.BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Set up the transcription configuration
config = assemblyai.TranscriptionConfig(audio_format='wav')

# Add a button to start and stop the recording
if st.button("Start recording"):
    # Start the recording
    recorder.start_recording()

if st.button("Stop recording"):
    # Stop the recording and get the recorded data
    data, samplerate = recorder.stop_recording()

    # Display the recorded audio
    st.audio(data, rate=samplerate)

    # Send the recorded audio to the AssemblyAI API for transcription
    transcription = client.transcribe(audio=data, config=config)

    # Tokenize the text
    input_ids = transformers.BertTokenizer.from_pretrained('bert-base-uncased').encode(transcription.text, return_tensors='pt')

    # Get the sentiment prediction
    prediction = model(input_ids)[0]

    # Get the sentiment label with the highest probability
    sentiment_label = prediction.argmax(dim=1).item()

    # Map the label to a sentiment
    if sentiment_label == 0:
        sentiment = "negative"
    elif sentiment_label == 1:
        sentiment = "neutral"
    else:
        sentiment = "positive"

    # Display the transcription and the sentiment
    st.write(f"Transcription: {transcription.text}")
    st.write(f"Sentiment: {sentiment}")

# Add a button to save the recorded audio to a file
if st.button("Save recording"):
    # Save the recorded audio to a WAV file
    sf.write("recording.wav", data, samplerate)
    st.success("Saved recording to file")
