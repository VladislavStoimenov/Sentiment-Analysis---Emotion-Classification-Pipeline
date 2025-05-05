import os
import torch
import whisper
import pandas as pd
from pytubefix import YouTube
from pydub import AudioSegment
import subprocess
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GenerationConfig, TFAutoModelForSeq2SeqLM
import tensorflow as tf
import numpy as np

# -----------------------------
# CONFIGURATION
# -----------------------------

YOUTUBE_LINK = "https://www.youtube.com/watch?v=8ZJHD7GECuM"
AUDIO_FILE = "audio.wav"
TRANSCRIPTION_FILE = "transcription.csv"
TRANSLATION_FILE = "translation.csv"
FINAL_OUTPUT_FILE = "pipeline_predictionsv2.csv"

TF_MODEL_PATH = "tf_model"
BERT_MODEL_PATH = "bert_large_8"

# -----------------------------
# UTILITY FUNCTIONS
# -----------------------------

def clean_text(text):
    """Removes leading/trailing quotes and whitespace from a sentence."""
    return text.strip().strip('"').strip("'")

def check_gpu():
    """Returns True if GPU is available for PyTorch or TensorFlow."""
    tf_gpu = tf.config.list_physical_devices('GPU')
    torch_gpu = torch.cuda.is_available()
    return tf_gpu or torch_gpu

# -----------------------------
# STEP 1: DOWNLOAD AUDIO
# -----------------------------


def download_audio(youtube_link, output_path):
    """
    Downloads YouTube video using pytubefix and converts audio to .wav format.

    Args:
        youtube_link (str): YouTube video URL.
        output_path (str): Path to save the final .wav file (e.g., 'audio.wav').
    """
    print("[Step 1] Downloading video and extracting audio...")

    yt = YouTube(youtube_link)
    stream = yt.streams.filter(only_audio=True).first()

    # Download as .mp4 (default behavior)
    downloaded_file = stream.download(filename="temp_audio.mp4")

    # Convert to .wav using pydub
    audio = AudioSegment.from_file(downloaded_file)
    audio.export(output_path, format="wav")

    # Clean up
    os.remove(downloaded_file)


# -----------------------------
# STEP 2: TRANSCRIBE AUDIO
# -----------------------------

def transcribe_audio(audio_path, output_csv):
    """Transcribes Bulgarian audio using Whisper-large and saves CSV with timestamps."""
    print("[Step 2] Transcribing audio...")
    model = whisper.load_model("large")

    result = model.transcribe(audio_path, language="bg")
    segments = result['segments']

    data = []
    for seg in segments:
        start = pd.to_datetime(seg['start'], unit='s').strftime("%H:%M:%S,%f")[:-3]
        end = pd.to_datetime(seg['end'], unit='s').strftime("%H:%M:%S,%f")[:-3]
        sentence = clean_text(seg['text'])
        data.append([start, end, sentence])

    df = pd.DataFrame(data, columns=["Start Time", "End Time", "Sentence"])
    df.to_csv(output_csv, index=False)

# -----------------------------
# STEP 3: TRANSLATE SENTENCES
# -----------------------------

def translate_sentences(input_csv, output_csv, tf_model_path):
    """
    Translates Bulgarian sentences to English using a HuggingFace MarianMT model.

    Args:
        input_csv (str): Path to CSV file containing a 'Sentence' column in Bulgarian.
        output_csv (str): Path to output CSV file with an added 'Translation' column.
        tf_model_path (str): Path to the HuggingFace-style folder containing the translation model, tokenizer, and config.
    """
    print("[Step 3] Translating sentences...")

    # Load model, tokenizer, and generation config
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-bg-en")
    model = TFAutoModelForSeq2SeqLM.from_pretrained(tf_model_path)
    gen_config = GenerationConfig.from_pretrained(tf_model_path)

    # Use GPU if available
    device = "/GPU:0" if tf.config.list_physical_devices("GPU") else "/CPU:0"

    df = pd.read_csv(input_csv)
    translations = []

    with tf.device(device):
        for sentence in df["Sentence"]:
            cleaned = clean_text(sentence)
            inputs = tokenizer(cleaned, return_tensors="tf", padding=True, truncation=True)
            outputs = model.generate(**inputs, **gen_config.to_dict())
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            translations.append(clean_text(translated))

    df["Translation"] = translations
    df.to_csv(output_csv, index=False)

# -----------------------------
# STEP 4: EMOTION CLASSIFICATION
# -----------------------------

def classify_emotions(input_csv, output_csv, bert_model_path):
    """Classifies emotions using a fine-tuned BERT model."""
    print("[Step 4] Predicting emotions...")

    df = pd.read_csv(input_csv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(bert_model_path)
    model.to(device)
    model.eval()

    emotions = []
    for sent in df["Translation"]:
        inputs = tokenizer(clean_text(sent), return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=1).item()
        label_map = model.config.id2label if hasattr(model.config, "id2label") else {i: str(i) for i in range(model.config.num_labels)}
        emotions.append(label_map[prediction])

    df["Emotion"] = emotions
    df.to_csv(output_csv, index=False, encoding="utf-8")

# -----------------------------
# MAIN SCRIPT
# -----------------------------

if __name__ == "__main__":
    print("========== Starting Pipeline ==========")

    # STEP 1: AUDIO
    if not os.path.exists(AUDIO_FILE):
        download_audio(YOUTUBE_LINK, AUDIO_FILE)
    else:
        print("[Skip] Audio already exists.")

    # STEP 2: TRANSCRIPTION
    if not os.path.exists(TRANSCRIPTION_FILE):
        transcribe_audio(AUDIO_FILE, TRANSCRIPTION_FILE)
    else:
        print("[Skip] Transcription already exists.")

    # STEP 3: TRANSLATION
    if not os.path.exists(TRANSLATION_FILE):
        translate_sentences(TRANSCRIPTION_FILE, TRANSLATION_FILE, TF_MODEL_PATH)
    else:
        print("[Skip] Translation already exists.")

    # STEP 4: EMOTION CLASSIFICATION
    if not os.path.exists(FINAL_OUTPUT_FILE):
        classify_emotions(TRANSLATION_FILE, FINAL_OUTPUT_FILE, BERT_MODEL_PATH)
    else:
        print("[Skip] Emotion predictions already exist.")

    print("Pipeline complete. Output saved to:", FINAL_OUTPUT_FILE)
