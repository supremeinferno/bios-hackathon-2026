# Data deletion checklist (FORGE Stage 14)

The data-use rules (PDF p.2) require deleting **every copy** of the footage when the event ends,
including uploads to Colab, Kaggle and API file storage. Breaking the rules means disqualification.
Tick every line; nothing here contains data.

## On every team laptop (repo root)

| Copy | Path | Command (Git Bash) |
|---|---|---|
| Original clips | `data/raw/` | `rm -rf data/raw` |
| Uploaded clips (dashboard) | `data/uploads/` | `rm -rf data/uploads` |
| Sanitized copies | `data/sanitized/` | `rm -rf data/sanitized` |
| Trimmed copies | `data/trimmed/` | `rm -rf data/trimmed` |
| Frame cache | `cache/frames/` | `rm -rf cache` |
| API upload copies + responses | `cache/api/` | (included above) |
| Pipeline outputs (hand tracks etc. are derived data) | `outputs*/` | `rm -rf outputs outputs_*` |
| Frames, contact sheets, demo renders | `artifacts/` | `rm -rf artifacts` |
| Zip archives of the data | e.g. `data.zip` | `rm -f data.zip` |
| Browser downloads of results / demo video | `~/Downloads/` | delete by hand |
| Recycle Bin | | empty it after deleting |

Check nothing is left: `git status --ignored` should list no data folders.

## Cloud

- [ ] **Gemini file storage:** run the snippet below; it must print `0`. (The pipeline deletes every upload
  in a `finally` block, and we have checked this after each failed run, but check once more at the end.)
  ```bash
  .venv/Scripts/python -c "from dotenv import dotenv_values; from google import genai; c=genai.Client(api_key=dotenv_values('.env')['GEMINI_API_KEY']); [c.files.delete(name=f.name) for f in c.files.list()]; print(len(list(c.files.list())))"
  ```
- [ ] **Google Drive:** demo videos and slides were uploaded only to the organisers' shared folder.
  Remove any personal copies.
- [ ] **Colab / Kaggle:** not used by this project. If anyone used them, delete notebooks' data and outputs.
- [ ] **OneDrive:** this repo lives under OneDrive\Desktop. If sync was ever on, delete the cloud copies
  of `data/`, `outputs*/`, `artifacts/`, `cache/` and `data.zip` from onedrive.com as well, and empty
  the OneDrive recycle bin.

## Keys

- [ ] Revoke or rotate the Gemini API key in Google AI Studio, then delete `.env`.

## Sign-off

| Person | Laptop | Date | All lines ticked |
|---|---|---|---|
| | | | |
