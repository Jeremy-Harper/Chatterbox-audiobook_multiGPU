# This is a modified version of [Chatterbox TTS](https://huggingface.co/ResembleAI/chatterbox). It supports multiple GPU's on a system and splits the work into a queue.

# 🚀 Chatterbox-TTS-Extended — Features & Explanations

Chatterbox-TTS-Extended is an advanced, highly customizable text-to-speech (TTS) interface built on [Chatterbox-TTS](https://github.com/resemble-ai/chatterbox), offering powerful options for voice, batching, post-processing, file handling, and artifact reduction.

## 📋 Table of Contents

- [Text & Input Features](#text--input-features)
- [Voice & Synthesis Controls](#voice--synthesis-controls)
- [Batching & Chunking](#batching--chunking)
- [Audio Post-Processing](#audio-post-processing)
- [Candidate & Validation Logic](#candidate--validation-logic)
- [Output & Export Options](#output--export-options)
- [Sound Word Replacement/Removal](#sound-word-replacementremoval)
- [Parallel Processing & Performance](#parallel-processing--performance)
- [Whisper Sync & Validation](#whisper-sync--validation)
- [Other Features](#other-features)
- [Tips & Troubleshooting](#tips--troubleshooting)
- [Installation](#installation)

---

## Text & Input Features

### **Text Input**
- **Direct Input:** Enter text directly in the textbox for synthesis.
- **Multi-file Upload:** Upload one or more `.txt` files. When multiple are uploaded, you can combine or process them individually.

### **Generate Separate Audio Files Per Text File**
- **Batch Mode:** Optionally, each uploaded text file produces a separate output file.
- **Merge Mode:** All uploaded files are combined (alphabetically) into a single synthesized audio.

### **Reference Audio (Voice Prompt)**
- **Reference Audio:** Upload or record a short sample to use as a style or voice prompt. The TTS model attempts to mimic this reference.

---

## Voice & Synthesis Controls

### **Emotion Exaggeration**
- Controls how dramatically emotions (excited, sad, angry, etc.) are expressed in speech.
- Range: `0.0` (neutral) to `2.0` (highly exaggerated). `1.0` is the model's standard expressiveness.

### **CFG Weight (Classifier-Free Guidance)**
- Balances strictness to input text vs. naturalness in output.
- High values (`1.0`) = more literal, less expressive.
- Low values = more natural, potentially less precise to the original text.

### **Temperature**
- Adds controlled randomness to the TTS generation.
- Low values = more predictable, consistent speech.
- High values = more variety and unpredictability.

### **Random Seed**
- `0` = random each time (unique outputs).
- Any other integer = repeatable, for reproducibility.

---

## Batching & Chunking

### **Sentence Batching**
- Automatically groups sentences into "chunks" (up to a character limit, e.g., 400 chars) for smoother, more natural speech.
- Reduces awkward pauses and enables faster, parallel processing.

### **Smart-Append Short Sentences**
- When batching is off, very short sentences are merged with their neighbors to avoid unnatural choppiness.

### **Convert 'J.R.R.' Style Input**
- Detects abbreviations written with periods and spaces them out (e.g., `J.R.R.` ➔ `J R R`) for correct pronunciation.

---

## Audio Post-Processing

### **Auto-Editor Integration**
- Uses [auto-editor](https://github.com/WyattBlue/auto-editor) to remove silence and TTS artifacts.
- **Volume Threshold:** Sets minimum loudness to consider as speech.
- **Margin:** Buffer time before and after detected speech to prevent cutting off words or breaths.
- **Keep Original:** Optionally retains the raw, unprocessed WAV alongside the cleaned version.

### **Audio Normalization (ffmpeg)**
- Uses `ffmpeg` for post-processing:
  - **EBU R128 Loudness Normalization:** Adjusts to a target loudness (TV/podcast standard).
  - **Peak Normalization:** Ensures audio peaks don't clip.

### **Customizable Normalization Settings**
- **Integrated Loudness (I):** Target perceived loudness.
- **True Peak (TP):** Max allowable peak.
- **Loudness Range (LRA):** Controls compression/dynamics.

---

## Candidate & Validation Logic

### **Number of Generations**
- Specify how many different outputs ("takes") to generate for the same input.

### **Number of Candidates Per Chunk**
- For each chunk, generate several variants and select the best one based on validation logic.
- Helps reduce TTS artifacts and find the best-sounding output.

### **Max Attempts Per Candidate**
- How many times to retry a candidate if it fails validation (up to N retries per candidate).

### **Bypass Whisper Checking**
- If enabled, candidate validation (via Whisper) is skipped; the shortest-duration candidate is selected (faster but riskier).

### **Fallback Logic When All Candidates Fail**
- If all candidates fail validation, can pick either:
  - The candidate with the highest Whisper similarity score (best match to input text)
  - Or, if enabled, the one with the longest transcribed text (useful for highly variable outputs)

---

## Output & Export Options

### **Export Formats**
- **Multi-format Output:** Choose one or more output types: `wav`, `mp3` (high quality, 320k), `flac`.
- **Export all selected formats at once.**

### **Output File Naming**
- Filenames include a base name (from uploaded file), timestamp, generation number, and seed for easy tracking.

---

## Sound Word Replacement/Removal

### **Sound Words / Word Replacement**
- **Remove or Replace Words:** Supply a list of "problem words" to remove or substitute before synthesis.
- **Format:**  
  - Remove: `ss, sss, hmm` (comma/newline separated)
  - Replace: `Baggins=>Gomberg`
  - Handles possessives, quoted words, and standalone occurrences.

---

## Parallel Processing & Performance

### **Parallel Chunk Processing**
- Synthesizes multiple chunks at once for huge speed gains on capable GPUs.
- **Parallel Workers:** Set the number of concurrent jobs (1 = sequential, higher = more parallelism).

---

## Whisper Sync & Validation

### **Whisper Model Selection**
- Select which OpenAI Whisper model to use for transcription validation (`tiny`, `base`, `small`, `medium`, `large`), with estimated VRAM usage.
- Smaller models are faster/less accurate, large ones are slower/more precise.

### **Whisper-Based Validation**
- Each audio candidate is transcribed using Whisper and checked against the original input.
- The best candidate is selected by highest match score (or shortest duration if Whisper is bypassed).
---

## Other Features

### **Text Preprocessing**
- **Convert to Lowercase:** Normalize input for consistency.
- **Normalize Whitespace:** Remove redundant spaces and blank lines.

### **Debug Output**
- Extensive terminal debug logging with color coding for progress, warnings, and errors.

### **Error Handling**
- Attempts to catch and report errors throughout TTS generation, candidate validation, file I/O, and post-processing steps.

---

## Tips & Troubleshooting

- **VRAM or speed issues?**  
  - Reduce parallel workers
  - Switch to a smaller Whisper model
  - Lower number of candidates per chunk

- **Audio choppy or abrupt?**  
  - Increase Auto-Editor margin
  - Lower silence threshold

- **Want consistent results?**  
  - Set a non-zero random seed

- **Artifacts or mispronunciations?**  
  - Use more candidates, increase attempts, or adjust sound word replacements

---

## 📝 License & Credits

See the main repo for license details.  
Original TTS: [Chatterbox-TTS](https://github.com/resemble-ai/chatterbox)  
Auto-Editor: [WyattBlue/auto-editor](https://github.com/WyattBlue/auto-editor)  
Whisper: [OpenAI Whisper](https://github.com/openai/whisper)

---

## 📣 Feedback & Contributions

Open an issue or pull request with your suggestions, bug reports, or improvements!

---

# Installation

I'm running this in a Python 3.10.6 virtual environment. I do not know what other versions work.

Clone the repo
`git clone https://github.com/petermg/Chatterbox-TTS-Extended`

Then install via
`pip install --force-reinstall -r requirements.txt`  

<sup> if for some reason the install doesn't run try doing </sup> `pip install --force-reinstall -r requirements.base.with.versions.txt`, 
<sup> and if that still doesn't work then do </sup> `pip install --force-reinstall -r requirements_frozen.txt`

Then run via
`python Chatter.py`


[FFMPEG](https://ffmpeg.org/download.html) is required. If you don't have it installed in your system path, put it in the same directory as the Chatter.py script.
