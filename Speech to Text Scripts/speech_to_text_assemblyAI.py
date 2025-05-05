import assemblyai as aai
import pandas as pd
import re
import sys

aai.settings.api_key = "0de7104d5dda4ef2bca9571d2eaab4d7"




def transcribe(file_url, language_code="bg"):
    transcriber = aai.Transcriber()
    config = aai.TranscriptionConfig(language_code=language_code, speech_model="nano")
    transcript = transcriber.transcribe(file_url, config=config)
    return transcript.text

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def save_transcript(sentences, output_file="transcribed_data_assemblyAI.csv"):
    df = pd.DataFrame(sentences, columns=["Sentence"])
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"Transcript saved to {output_file}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python speech_to_text.py <audio_file>")
        sys.exit(1)

    file_path = sys.argv[1]
    print(f'processing {file_path}')

    transcript_text = transcribe(file_path, language_code="bg")
    print("Transcription completed")

    sentences = split_into_sentences(transcript_text)
    save_transcript(sentences)

if __name__ == "__main__":
    main()