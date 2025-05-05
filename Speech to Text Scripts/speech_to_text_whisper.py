import whisper
import csv
from tqdm import tqdm

# Load the Whisper model
model = whisper.load_model("large").to("cuda")

# Define the audio file
audio_file = "audio.mp3"

# Transcribe audio with manual progress tracking
print("Transcription in progress...")

# Process the audio
result = model.transcribe(audio_file, language="bg")

# Save transcription as TXT
txt_file = audio_file.replace(".mp3", ".txt")
with open(txt_file, "w", encoding="utf-8") as f:
    f.write(result["text"])

# Save transcription as CSV with timestamps
csv_file = audio_file.replace(".mp3", ".csv")
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Start Time", "End Time", "Transcription"])

    for segment in result["segments"]:
        start_time = f"{segment['start']:.3f}"
        end_time = f"{segment['end']:.3f}"
        text = segment["text"]
        writer.writerow([start_time, end_time, text])

print(f"\n✅ Transcription saved to {txt_file} and {csv_file}")