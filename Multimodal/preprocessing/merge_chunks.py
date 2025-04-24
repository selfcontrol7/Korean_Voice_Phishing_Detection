import os
import argparse
from pydub import AudioSegment
from tqdm import tqdm

def merge_audio_chunks(root_folder, delete_chunks=False, output_format="wav"):
    conversations = []
    for dirpath, dirnames, filenames in os.walk(root_folder): # Traverse the directory tree
        wav_files = sorted([f for f in filenames if f.endswith('.wav')])
        if wav_files:
            conversations.append((dirpath, wav_files))
    # print the total number of subfolders
    print(f"Found {len(conversations)} subfolders with audio chunks.")
    print("-" * 50)

    for conv_path, chunks in tqdm(conversations, desc="Merging Conversations"):
        folder_name = os.path.basename(conv_path)
        combined = AudioSegment.empty()
        for chunk in chunks:
            chunk_path = os.path.join(conv_path, chunk)
            audio = AudioSegment.from_wav(chunk_path)
            combined += audio

        # Export merged audio with folder name
        # output_path = os.path.join(conv_path, f"{folder_name}.{output_format}")
        # print(f"Exporting {output_path}")

        # Export merged audio to data/audio/non_vishing folder with folder name
        export_folder = os.path.join("data", "audio", "non_vishing")
        os.makedirs(export_folder, exist_ok=True)
        # folder_name = os.path.basename(conv_path)
        relative_path = os.path.relpath(conv_path, root_folder)
        new_filename = "".join(relative_path.split(os.sep)).lower()
        output_path = os.path.join(export_folder, f"{new_filename}.{output_format}")
        print(f"Exporting {output_path}")

        combined.export(output_path, format=output_format)

        # Delete original chunks if flag is set
        if delete_chunks:
            for chunk in chunks:
                os.remove(os.path.join(conv_path, chunk))

def main():
    parser = argparse.ArgumentParser(description='Merge audio chunks into full conversation audio files named after their folder.')
    parser.add_argument('--input_folder', required=True, help='Root folder containing conversation subfolders')
    parser.add_argument('--delete_chunks', action='store_true', help='Delete original audio chunks after merging')
    parser.add_argument('--output_format', default='wav', help='Audio format for merged file (e.g., wav, mp3)')

    args = parser.parse_args()

    print(f"\nMerging audio chunks in: {args.input_folder}")
    merge_audio_chunks(args.input_folder, args.delete_chunks, args.output_format)
    print("\nDone! All conversations merged.")

if __name__ == "__main__":
    main()
