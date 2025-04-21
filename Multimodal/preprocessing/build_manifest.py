import os
import json

def write_master_manifest(audio_root: str,
                          transcript_root: str,
                          output_path: str):
    """
    Scans audio_root/{vishing,non_vishing} for audio files,
    finds transcripts under transcript_root/{same_subdir}/{call_id}.json,
    and writes a JSONL manifest with call_id, file paths, and label.
    """
    manifest_entries = []

    for subdir, label in [('vishing', 1), ('non_vishing', 0)]:
        audio_folder = os.path.join(audio_root, subdir)
        transcript_folder = os.path.join(transcript_root, subdir)

        if not os.path.isdir(audio_folder):
            raise FileNotFoundError(f"Audio folder not found: {audio_folder}")
        if not os.path.isdir(transcript_folder):
            raise FileNotFoundError(f"Transcript folder not found: {transcript_folder}")

        for fname in sorted(os.listdir(audio_folder)):
            if not fname.lower().endswith(('.wav', '.mp3', '.mp4')):
                continue

            call_id, _ = os.path.splitext(fname)
            audio_fp = os.path.join(audio_folder, fname)
            transcript_fp = os.path.join(transcript_folder, f"{call_id}.json")

            if not os.path.exists(transcript_fp):
                print(f"[!] No transcript for {call_id} in {transcript_folder}, skipping.")
                continue

            manifest_entries.append({
                "call_id": call_id,
                "audio_filepath": audio_fp,
                "transcript_filepath": transcript_fp,
                "label": label
            })

    # check if the manifest file already exists
    if os.path.exists(output_path):
        print(f"[!] Manifest file already exists at {output_path}.")
        # ask the user if they want to delete it
        delete_manifest = input("Do you want to delete it? (y/n): ").strip().lower()
        if delete_manifest == 'y':
            os.remove(output_path)
            print(f"Manifest file at {output_path} deleted.")
        else:
            print("Exiting without writing a new manifest.")
            # overwrite the existing manifest
            return

    with open(output_path, "w", encoding="utf-8") as out_f:
        for entry in manifest_entries:
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    # adjust these paths to your layout:
    AUDIO_ROOT = "../data/audio/"
    # under AUDIO_ROOT you must have `vishing/` and `non_vishing/`
    TRANSCRIPT_ROOT = "../data/transcripts/"
    OUTPUT = "../data/master_manifest.jsonl"

    write_master_manifest(AUDIO_ROOT, TRANSCRIPT_ROOT, OUTPUT)
    print(f"✅ Manifest written to {OUTPUT}")
