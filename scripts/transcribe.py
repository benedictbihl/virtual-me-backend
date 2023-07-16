import os

from tqdm import tqdm
from faster_whisper import WhisperModel
from dotenv import load_dotenv

load_dotenv()

model = WhisperModel("medium", device="cpu", compute_type="int8")

# Define the folder where the wav files are located
root_folder = os.getenv("ROOT_DIR") or ""
print("Root folder: ", root_folder)


# Get the number of wav files in the root folder and its sub-folders
print("Getting number of files to transcribe...")
num_files = sum(
    1
    for dirpath, dirnames, filenames in os.walk(root_folder)
    for filename in filenames
    if filename.endswith(".m4a")
)
print("Number of files: ", num_files)

# Transcribe the wav files and display a progress bar
with tqdm(total=num_files, desc="Transcribing Files") as pbar:
    for dirpath, dirnames, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".m4a"):
                filepath = os.path.join(dirpath, filename)
                audio_file = open(root_folder + filename, "rb")
                segments, info = model.transcribe(
                    audio_file, beam_size=5, language="en"
                )

                # print(
                #     "Detected language '%s' with probability %f"
                #     % (info.language, info.language_probability)
                # )

                # Write transcription to text file
                filename_no_ext = os.path.splitext(filename)[0]
                folder_path = os.path.join(dirpath, "transcripts")

                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                with open(
                    os.path.join(folder_path, filename_no_ext + ".txt"),
                    "w",
                ) as f:
                    for segment in segments:
                        f.write(segment.text)
                pbar.update(1)
