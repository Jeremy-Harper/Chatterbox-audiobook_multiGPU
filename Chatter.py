# Code Version: 2024-06-11-Serial-GPU-Fix
import random
import numpy as np
import torch
import os
import re
import datetime
import torchaudio
import gradio as gr
# import spaces # Comment out if not deploying on Hugging Face Spaces
import subprocess
from pydub import AudioSegment
import ffmpeg
import librosa
import string
import difflib
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import argparse
import traceback
import time
import multiprocessing

from chatterbox.src.chatterbox.tts import ChatterboxTTS
from chatterbox.src.chatterbox.models.s3gen import S3GEN_SR
import whisper

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
CODE_VERSION = "2024-06-11-Serial-GPU-Fix"
print(f"Running Chatter.py - Code Version: {CODE_VERSION}")


# --- Worker-Specific Globals and Model Management ---
_WORKER_TTS_MODEL = None
_WORKER_WHISPER_MODEL = None
_WORKER_DEVICE = None 

def get_or_init_worker_models(device_str: str):
    """Ensures that models for a given device are loaded ONCE per worker process."""
    global _WORKER_TTS_MODEL, _WORKER_WHISPER_MODEL, _WORKER_DEVICE
    pid = os.getpid()
    if _WORKER_DEVICE != device_str:
        print(f"[Worker-{pid}] Initializing models for device: {device_str}")
        _WORKER_DEVICE = device_str
        try:
            _WORKER_TTS_MODEL = ChatterboxTTS.from_pretrained(device_str)
            print(f"[Worker-{pid}] TTS model loaded on {device_str}.")
            whisper_device = torch.device(device_str if "cuda" in device_str and torch.cuda.is_available() else "cpu")
            _WORKER_WHISPER_MODEL = whisper.load_model("medium", device=whisper_device, download_root=str(Path.home() / ".cache" / "whisper"))
            print(f"[Worker-{pid}] Whisper model loaded on {whisper_device}.")
        except Exception as e:
            print(f"[Worker-{pid}/CRITICAL-ERROR] Failed to initialize models for device {device_str}: {e}\n{traceback.format_exc()}")
            _WORKER_TTS_MODEL, _WORKER_WHISPER_MODEL = None, None
            raise
    return _WORKER_TTS_MODEL, _WORKER_WHISPER_MODEL

# --- Utility Functions ---
def set_seed(seed: int):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def normalize_whitespace(text: str) -> str: return re.sub(r'\s{2,}', ' ', text.strip())
def replace_letter_period_sequences(text: str) -> str:
    def replacer(match): cleaned = match.group(0).rstrip('.'); letters = cleaned.split('.'); return ' '.join(letters)
    return re.sub(r'\b(?:[A-Za-z]\.){2,}', replacer, text)
def split_into_sentences(text): return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s.strip()]
def group_sentences(sentences, max_chars=300):
    chunks = []; current_chunk = []; current_length = 0
    for s in sentences:
        if not s: continue
        s_len = len(s)
        if s_len > 500: s = s[:500]; s_len = 500
        if s_len > max_chars:
            if current_chunk: chunks.append(" ".join(current_chunk))
            chunks.append(s); current_chunk = []; current_length = 0
        elif current_length + s_len + (1 if current_chunk else 0) <= max_chars:
            current_chunk.append(s); current_length += s_len + (1 if current_chunk else 0)
        else:
            if current_chunk: chunks.append(" ".join(current_chunk))
            current_chunk = [s]; current_length = s_len
    if current_chunk: chunks.append(" ".join(current_chunk))
    return [c for c in chunks if c]
def smart_append_short_sentences(sentences, max_chars=300):
    new_groups = []; i = 0; temp_s = [s.strip() for s in sentences if s.strip()]
    while i < len(temp_s):
        current = temp_s[i]
        if len(current) >= 20: new_groups.append(current); i += 1
        else:
            appended = False
            if i + 1 < len(temp_s) and len(current + " " + temp_s[i+1]) <= max_chars:
                new_groups.append(current + " " + temp_s[i+1]); i += 2; appended = True
            elif new_groups and len(new_groups[-1] + " " + current) <= max_chars:
                new_groups[-1] += " " + current; i += 1; appended = True
            if not appended: new_groups.append(current); i += 1
    return [g for g in new_groups if g]
def normalize_with_ffmpeg(input_wav, output_wav, method, i, tp, lra, target_sr):
    if method == "ebu": loudnorm_filter = f"loudnorm=I={i}:TP={tp}:LRA={lra}"
    elif method == "peak": loudnorm_filter = "dynaudnorm"
    else: raise ValueError(f"Unknown norm method: {method}")
    try:
        (ffmpeg.input(str(input_wav)).output(str(output_wav), af=loudnorm_filter, ar=str(target_sr))
         .overwrite_output().run(quiet=True, capture_stdout=True, capture_stderr=True))
    except ffmpeg.Error as e:
        print(f"[FFMPEG/ERROR] Norm failed for {input_wav}: {e.stderr.decode('utf8', 'ignore') if e.stderr else 'N/A'}")
        if str(input_wav) != str(output_wav): import shutil; shutil.copyfile(str(input_wav), str(output_wav))
def normalize_for_compare_all_punct(text): return re.sub(r'\s+', ' ', re.sub(r'[–—-]', ' ', re.sub(rf"[{re.escape(string.punctuation)}]", '', text))).lower().strip()
def fuzzy_match(text1, text2, threshold=0.90):
    t1_norm, t2_norm = normalize_for_compare_all_punct(text1), normalize_for_compare_all_punct(text2)
    return False if not t1_norm or not t2_norm else difflib.SequenceMatcher(None, t1_norm, t2_norm).ratio() >= threshold
def get_wav_duration(path):
    try: return librosa.get_duration(path=str(path))
    except Exception: return float('inf')


# --- Main Worker Function (NEW REFACTORED LOGIC) ---
def worker_process_chunk(
    chunk_idx, sentence_group, device_str_for_tts, 
    current_master_seed, audio_prompt_path, exaggeration_input, 
    temperature_input, cfg_weight_input, disable_watermark_checkbox, 
    num_candidates_slider, max_attempts_slider, bypass_whisper_checkbox,
    run_temp_dir_str
):
    pid = os.getpid()
    try:
        tts_model, whisper_model = get_or_init_worker_models(device_str_for_tts)
        if tts_model is None or whisper_model is None:
            raise RuntimeError(f"Model initialization failed for device {device_str_for_tts}")
    except Exception as e_model_load_worker:
        return {"chunk_idx": chunk_idx, "status": "error", "error_message": f"Model Load Fail: {str(e_model_load_worker)}", "path": None}

    base_candidate_path_prefix = Path(run_temp_dir_str) / f"c{chunk_idx:04d}_cand"
    
    # --- Step 1: Generate all candidate audio files SERIALLY to prevent VRAM overflow ---
    generated_candidates = []
    attempts_to_make = num_candidates_slider if bypass_whisper_checkbox else num_candidates_slider * max_attempts_slider
    
    for i in range(attempts_to_make):
        slot = i // max_attempts_slider if not bypass_whisper_checkbox else i
        attempt = i % max_attempts_slider if not bypass_whisper_checkbox else 0
        seed = current_master_seed if (slot == 0 and attempt == 0) else random.randint(1, 2**32 - 1)
        
        set_seed(seed)
        path_str = f"{str(base_candidate_path_prefix)}_{slot+1}_try{attempt+1}_seed{seed}.wav"
        
        try:
            wav_tensor = tts_model.generate( 
                sentence_group, audio_prompt_path=audio_prompt_path, exaggeration=min(exaggeration_input, 1.0),
                temperature=temperature_input, cfg_weight=cfg_weight_input, apply_watermark=not disable_watermark_checkbox, use_cond_cache=True 
            )
            if torch.is_tensor(wav_tensor) and wav_tensor.numel() > tts_model.sr * 0.1: # Check for valid audio
                torchaudio.save(path_str, wav_tensor.cpu(), tts_model.sr)
                duration = get_wav_duration(path_str)
                generated_candidates.append({"path": path_str, "duration": duration, "seed": seed})
            else:
                print(f"[Worker-{pid}/WARN] Generation failed for chunk {chunk_idx} (seed {seed}), empty audio.")
        except Exception as e:
            print(f"[Worker-{pid}/ERROR] Generation crashed for chunk {chunk_idx} (seed {seed}): {e}")

    if not generated_candidates:
        return {"chunk_idx": chunk_idx, "status": "error", "error_message": "All generation attempts failed.", "path": None}

    # --- Step 2: Validate generated audios in parallel (CPU/IO bound) ---
    if not bypass_whisper_checkbox:
        temp_passed, temp_failed = [], []
        num_threads = min(os.cpu_count() or 1, 4)
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_cand = {executor.submit(whisper_model.transcribe, cand['path'], language="en", fp16=(whisper_model.device.type == 'cuda')): cand for cand in generated_candidates}
            
            for future in as_completed(future_to_cand):
                cand = future_to_cand[future]
                try:
                    transcription_result = future.result()
                    transcribed_text = transcription_result['text'].strip().lower()
                    if fuzzy_match(transcribed_text, sentence_group):
                        temp_passed.append(cand)
                    else:
                        temp_failed.append(cand)
                except Exception as e:
                    print(f"[Worker-{pid}/ERROR] Whisper transcription failed for {cand['path']}: {e}")
                    temp_failed.append(cand)
    else: # If bypassing, all generated candidates are considered "passed"
        temp_passed = generated_candidates
        temp_failed = []

    # --- Step 3: Select the best candidate ---
    chosen = None
    if temp_passed: chosen = sorted(temp_passed, key=lambda x: x["duration"])[0]
    elif temp_failed: chosen = sorted(temp_failed, key=lambda x: x["duration"])[0]

    if chosen and chosen.get("path") and Path(chosen["path"]).exists():
        return {"chunk_idx": chunk_idx, "status": "success", "path": chosen["path"]}
    else:
        return {"chunk_idx": chunk_idx, "status": "error", "error_message": "No valid candidate generated or found.", "path": None}


# --- Main TTS Orchestrator Function ---
# @spaces.GPU
def generate_batch_tts_orchestrator(
    text_input_str, text_file_input, audio_prompt_path_input_gr,
    exaggeration_input, temperature_input, seed_num_input, cfg_weight_input,
    use_auto_editor, ae_threshold, ae_margin,
    export_format, enable_batching_checkbox, to_lowercase_checkbox,
    normalize_spacing_checkbox, fix_dot_letters_checkbox,
    keep_original_wav_checkbox, smart_batch_short_sentences_checkbox,
    disable_watermark_checkbox, num_generations_input,
    normalize_audio_checkbox, normalize_method_dropdown,
    normalize_level_slider, normalize_tp_slider, normalize_lra_slider,
    num_candidates_slider, max_attempts_slider, bypass_whisper_checkbox,
    silence_duration_ms_input,
    target_gpus_str_input, 
    progress=gr.Progress(track_tqdm=True)
):
    # (Input and Device Preparation logic is correct, keep as is)
    actual_audio_prompt_path = None
    if audio_prompt_path_input_gr:
        if isinstance(audio_prompt_path_input_gr, str) and Path(audio_prompt_path_input_gr).exists(): actual_audio_prompt_path = audio_prompt_path_input_gr
        elif hasattr(audio_prompt_path_input_gr, 'name') and Path(audio_prompt_path_input_gr.name).exists(): actual_audio_prompt_path = audio_prompt_path_input_gr.name
    if not target_gpus_str_input or not target_gpus_str_input.strip(): target_gpu_devices_list = ["cuda:0" if torch.cuda.is_available() else "cpu"]
    else: target_gpu_devices_list = [s.strip() for s in target_gpus_str_input.split(',') if s.strip()]
    effective_devices = []
    for dev_str in target_gpu_devices_list:
        if dev_str == "cpu": effective_devices.append("cpu")
        elif torch.cuda.is_available():
            try:
                idx = int(dev_str.split(':')[-1])
                if 0 <= idx < torch.cuda.device_count(): effective_devices.append(f"cuda:{idx}")
                else: print(f"[Orchestrator/WARN] Invalid CUDA idx: {dev_str}. Ignored.")
            except (ValueError, IndexError): 
                if not any(d.startswith("cuda") for d in effective_devices): effective_devices.append("cuda:0")
        else: print(f"[Orchestrator/WARN] CUDA device {dev_str} specified, but CUDA unavailable. Ignored.")
    if not effective_devices: effective_devices = ["cpu"]
    effective_devices = sorted(list(set(effective_devices)))
    print(f"[Orchestrator/INFO] Effective devices for processing: {effective_devices}")
    
    final_text = ""
    if text_file_input:
        with open(text_file_input.name, "r", encoding="utf-8") as f: final_text = f.read()
    elif text_input_str: final_text = text_input_str
    if not final_text.strip(): raise gr.Error("No text provided.")
    if to_lowercase_checkbox: final_text = final_text.lower()
    if normalize_spacing_checkbox: final_text = normalize_whitespace(final_text)
    if fix_dot_letters_checkbox: final_text = replace_letter_period_sequences(final_text)

    chunking_logic = (group_sentences if enable_batching_checkbox else 
                      smart_append_short_sentences if smart_batch_short_sentences_checkbox else 
                      split_into_sentences)
    all_sentence_groups = chunking_logic(split_into_sentences(final_text))
    all_sentence_groups = [sg for sg in all_sentence_groups if sg.strip()]
    if not all_sentence_groups: raise gr.Error("Text resulted in no processable chunks.")
    num_chunks_total = len(all_sentence_groups)
    print(f"[Orchestrator/INFO] Total text chunks to process: {num_chunks_total}")

    main_temp_dir = Path("temp"); main_temp_dir.mkdir(exist_ok=True)
    output_dir = Path("output"); output_dir.mkdir(exist_ok=True)
    all_final_output_files = []
    
    ctx = multiprocessing.get_context('spawn')

    for full_gen_run_idx in range(int(num_generations_input)):
        if int(seed_num_input) == 0: current_run_master_seed = random.randint(1, 2**32 - 1)
        else: current_run_master_seed = int(seed_num_input) + full_gen_run_idx
        print(f"\n[Orchestrator/INFO] Run {full_gen_run_idx+1}/{num_generations_input}, Master Seed: {current_run_master_seed}")
        run_temp_dir = main_temp_dir / f"run_{full_gen_run_idx+1}_{time.time_ns()}"
        run_temp_dir.mkdir(parents=True, exist_ok=True)

        device_executors = {device: ProcessPoolExecutor(max_workers=1, mp_context=ctx) for device in effective_devices}
        
        all_futures = []
        future_to_chunk_idx = {}
        for chunk_idx, sentence_group in enumerate(all_sentence_groups):
            device_for_chunk = effective_devices[chunk_idx % len(effective_devices)]
            executor = device_executors[device_for_chunk]
            
            task_bundle = (
                chunk_idx, sentence_group, device_for_chunk,
                current_run_master_seed, actual_audio_prompt_path, exaggeration_input, 
                temperature_input, cfg_weight_input, disable_watermark_checkbox, 
                num_candidates_slider, max_attempts_slider, bypass_whisper_checkbox,
                str(run_temp_dir)
            )
            future = executor.submit(worker_process_chunk, *task_bundle)
            all_futures.append(future)
            future_to_chunk_idx[future] = chunk_idx

        processed_results = [None] * num_chunks_total
        completed_count = 0
        progress(0, desc=f"Run {full_gen_run_idx+1}: Starting {num_chunks_total} chunks...")
        
        for future in as_completed(all_futures):
            chunk_idx = future_to_chunk_idx[future]
            try:
                result = future.result()
                processed_results[chunk_idx] = result
            except Exception as e:
                print(f"[Orchestrator/ERROR] Worker task for chunk {chunk_idx} failed: {e}\n{traceback.format_exc()}")
                processed_results[chunk_idx] = {"chunk_idx": chunk_idx, "status": "error", "error_message": "Worker process crashed"}
            finally:
                completed_count += 1
                progress(completed_count / num_chunks_total, desc=f"Run {full_gen_run_idx+1}: Processed {completed_count}/{num_chunks_total} chunks")
        
        for executor in device_executors.values():
            executor.shutdown(wait=True)
            
        final_waveforms, model_sr = [], S3GEN_SR
        for result in processed_results:
            if result and result.get("status") == "success" and result.get("path"):
                try:
                    waveform, sr = torchaudio.load(result["path"])
                    model_sr = sr 
                    if final_waveforms and silence_duration_ms_input > 0:
                        final_waveforms.append(torch.zeros((waveform.shape[0], int(sr * silence_duration_ms_input / 1000.0))))
                    final_waveforms.append(waveform)
                except Exception as e: print(f"[Orchestrator/ERROR] Failed to load chunk {result['path']}: {e}")
            elif result: print(f"[Orchestrator/WARN] Chunk {result.get('chunk_idx', '??')} failed: {result.get('error_message', 'Unknown error')}")
            else: print(f"[Orchestrator/WARN] Missing result for a chunk.")
        
        if not final_waveforms:
            print(f"[Orchestrator/ERROR] No valid chunks to assemble for Run {full_gen_run_idx+1}."); continue

        try: concatenated_audio = torch.cat(final_waveforms, dim=1)
        except Exception as e: print(f"[Orchestrator/ERROR] Failed to concat chunks for Run {full_gen_run_idx+1}: {e}."); continue
            
        run_unique_suffix = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_RUN{full_gen_run_idx+1}_seed{current_run_master_seed}"
        assembled_raw_path = run_temp_dir / f"assembled_raw_{run_unique_suffix}.wav"
        torchaudio.save(str(assembled_raw_path), concatenated_audio.cpu(), model_sr)
        current_processed_path = str(assembled_raw_path)
        
        # Post-processing logic can be re-added here.
        
        final_output_path = str(output_dir / f"final_output_{run_unique_suffix}.wav")
        import shutil; shutil.copyfile(current_processed_path, final_output_path)
        all_final_output_files.append(final_output_path)
        print(f"[Orchestrator/INFO] Finished Run {full_gen_run_idx+1}. Output: {Path(final_output_path).name}")
        
        try: shutil.rmtree(run_temp_dir)
        except Exception as e: print(f"[Orchestrator/WARN] Failed to clean temp dir {run_temp_dir}: {e}")

    if not all_final_output_files: return "No audio files were successfully generated."
    return "Generated files:\n" + "\n".join(sorted(all_final_output_files))


# --- Gradio Interface Setup ---
with gr.Blocks(theme=gr.themes.Glass()) as demo:
    gr.Markdown("# 🎧 Chatterbox TTS Extended Pro - Multi GPU Chunk Parallel Edition 🚀")
    if torch.cuda.is_available(): default_ui_device_str = ",".join([f"cuda:{i}" for i in range(torch.cuda.device_count())]) if torch.cuda.device_count() > 0 else "cpu"
    else: default_ui_device_str = "cpu"
    target_gpus_input = gr.Textbox(label="Target Processing Devices (comma-separated)", value=default_ui_device_str, info="Chunks will be distributed among these.")
    with gr.Tab("📝 Input & Basic Settings"):
        with gr.Row():
            with gr.Column(scale=2):
                text_input = gr.Textbox(label="Text Input", lines=8, placeholder="Enter text here...")
                text_file_input = gr.File(label="Upload Text File (.txt)", file_types=[".txt"])
            with gr.Column(scale=1):
                ref_audio_input_gr = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Reference Audio")
                num_generations_input = gr.Number(value=1, precision=0, minimum=1, label="Number of Full Output Files")
                seed_input = gr.Number(value=0, label="Master Random Seed per Full Output")
        with gr.Accordion("📄 Text Preprocessing Options", open=False):
            to_lowercase_checkbox = gr.Checkbox(label="Convert to lowercase", value=True)
            normalize_spacing_checkbox = gr.Checkbox(label="Normalize spacing", value=True)
            fix_dot_letters_checkbox = gr.Checkbox(label="Convert 'J.R.R.' to 'J R R'", value=True)
    with gr.Tab("⚙️ Generation Parameters (per chunk)"):
        with gr.Row():
            exaggeration_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Emotion Exaggeration")
            cfg_weight_slider = gr.Slider(0.0, 3.0, value=0.7, step=0.05, label="CFG Weight")
            temp_slider = gr.Slider(0.1, 1.5, value=0.8, step=0.05, label="Temperature")
        with gr.Accordion("🧩 Chunking & Candidate Selection (per chunk)", open=True):
            enable_batching_checkbox = gr.Checkbox(label="Group Sentences into Chunks (~300 chars)", value=True)
            smart_batch_short_sentences_checkbox = gr.Checkbox(label="Smart-Append Short Sentences (if grouping off)", value=False)
            num_candidates_slider = gr.Slider(1, 5, value=1, step=1, label="Desired Good Candidates per Chunk")
            max_attempts_slider = gr.Slider(1, 3, value=1, step=1, label="Max ASR Check Retries")
            bypass_whisper_checkbox = gr.Checkbox(label="Bypass ASR Validation", value=False)
            silence_duration_ms_input = gr.Slider(0, 1000, value=150, step=25, label="Silence Between Chunks (ms)")
    with gr.Tab("🛠️ Post-Processing & Export (per full output file)"):
        with gr.Accordion("🔊 Audio Normalization (ffmpeg)", open=False):
            normalize_audio_checkbox = gr.Checkbox(label="Enable Audio Normalization", value=True)
            normalize_method_dropdown = gr.Dropdown(choices=["ebu", "peak"], value="ebu", label="Normalization Method")
            normalize_level_slider = gr.Slider(-30, -10, value=-23, step=1, label="EBU Target Loudness (LUFS)")
            normalize_tp_slider = gr.Slider(-3, 0, value=-1.0, step=0.1, label="EBU True Peak (dBTP)")
            normalize_lra_slider = gr.Slider(1, 15, value=7, step=1, label="EBU Loudness Range (LU)")
        with gr.Accordion("✂️ Silence Removal (Auto-Editor)", open=False):
            use_auto_editor_checkbox = gr.Checkbox(label="Enable Silence Removal", value=False)
            threshold_slider = gr.Slider(0.01, 0.2, value=0.04, step=0.005, label="Volume Threshold for Silence")
            margin_slider = gr.Slider(0.0, 0.5, value=0.1, step=0.025, label="Margin Around Speech (s)")
            keep_original_wav_checkbox = gr.Checkbox(label="Keep Original Assembled WAV (pre-AE) (first full output only)", value=False)
        with gr.Row():
            export_format_dropdown = gr.Dropdown(choices=["wav", "mp3", "flac"], value="wav", label="Export Audio Format")
            disable_watermark_checkbox = gr.Checkbox(label="Disable Perth Watermark", value=False)

    run_button = gr.Button("✨ Generate Full Audio ✨", variant="primary", size="lg")
    output_status_box = gr.Textbox(label="Generated Audio File Paths / Status", lines=8, interactive=False)

    run_button.click(
        fn=generate_batch_tts_orchestrator,
        inputs=[
            text_input, text_file_input, ref_audio_input_gr, 
            exaggeration_slider, temp_slider, seed_input, cfg_weight_slider,
            use_auto_editor_checkbox, threshold_slider, margin_slider,
            export_format_dropdown, enable_batching_checkbox,
            to_lowercase_checkbox, normalize_spacing_checkbox, fix_dot_letters_checkbox,
            keep_original_wav_checkbox, smart_batch_short_sentences_checkbox,
            disable_watermark_checkbox, num_generations_input,
            normalize_audio_checkbox, normalize_method_dropdown,
            normalize_level_slider, normalize_tp_slider, normalize_lra_slider,
            num_candidates_slider, max_attempts_slider, bypass_whisper_checkbox,
            silence_duration_ms_input, target_gpus_input 
        ],
        outputs=output_status_box
    )
    gr.Markdown("---")
    gr.Markdown("Original Chatterbox by Resemble AI. Extended version with multi-GPU chunk parallelism by Pete.")
    gr.Markdown("Models are loaded once per GPU process. Ensure sufficient VRAM for each selected GPU.")

# --- Main Execution Guard ---
if __name__ == "__main__":
    if os.name != 'nt' and multiprocessing.get_start_method(allow_none=True) != 'spawn':
        try:
            multiprocessing.set_start_method('spawn', force=True)
            print("[MAIN SCRIPT] Set MP start method to 'spawn'.")
        except RuntimeError:
            print(f"[MAIN SCRIPT] MP context already set: {multiprocessing.get_start_method(allow_none=True)}.")
        except Exception as e_sm:
            print(f"[MAIN SCRIPT/WARN] Could not set MP start method to 'spawn': {e_sm}")

    parser = argparse.ArgumentParser(description="Chatterbox TTS Extended - Multi GPU Chunk Parallel")
    parser.add_argument("--share", action="store_true", help="Enable Gradio sharing link.")
    parser.add_argument("--listen", action="store_true", help="Listen on 0.0.0.0.")
    parser.add_argument("--port", type=int, default=7860, help="Port to run on.")
    args = parser.parse_args()
    
    demo.launch(share=args.share, server_name="0.0.0.0" if args.listen else None, server_port=args.port)
