import os


def shorten_names():
    audio_dir = "../data/audio/vishing"
    transcript_dir = "../data/transcripts/vishing"
    counter = 1

    # List only files in audio_dir and sort them for consistency.
    audio_files = sorted(
        f for f in os.listdir(audio_dir) if os.path.isfile(os.path.join(audio_dir, f))
    )
    print(f"Found {len(audio_files)} audio files in {audio_dir}.")

    for audio_file in audio_files:
        audio_path = os.path.join(audio_dir, audio_file) # Full path to the audio file
        base_name, audio_ext = os.path.splitext(audio_file) # Split the file name into base name and extension

        # Construct the corresponding transcript file name
        transcript_file = base_name + ".json"
        transcript_path = os.path.join(transcript_dir, transcript_file)

        # Check if the transcript file exists before renaming
        if not os.path.exists(transcript_path):
            print(f"Transcript file not found for '{audio_file}'. Skipping this pair.")
            continue

        # New base name using the incremental counter
        new_base = f"vishing_{counter}"
        new_audio_file = new_base + audio_ext
        new_transcript_file = new_base + ".json"

        new_audio_path = os.path.join(audio_dir, new_audio_file)
        new_transcript_path = os.path.join(transcript_dir, new_transcript_file)

        # Rename the audio and transcript files
        os.rename(audio_path, new_audio_path)
        os.rename(transcript_path, new_transcript_path)

        print(
            f"Renamed '{audio_file}' and '{transcript_file}' to "
            f"'{new_audio_file}' and '{new_transcript_file}'"
        )
        print("+"* 150)
        counter += 1


if __name__ == "__main__":
    shorten_names()