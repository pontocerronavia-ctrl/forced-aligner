from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import ctc_segmentation
import json
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo una vez al arrancar
MODEL_ID = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2Model.from_pretrained(MODEL_ID)
model.eval()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/align")
async def align(
    audio: UploadFile = File(...),
    text: str = Form(...)
):
    # Guardar audio temporalmente
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        # Cargar y resamplear a 16kHz
        waveform, sr = torchaudio.load(tmp_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(dim=0)  # mono

        # Obtener emissions del modelo
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).last_hidden_state
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        # CTC segmentation
        words = text.strip().split()
        char_list = processor.tokenizer.convert_ids_to_tokens(
            range(processor.tokenizer.vocab_size)
        )
        
        config = ctc_segmentation.CtcSegmentationParameters()
        config.char_list = char_list
        config.index_duration = 0.02  # 20ms por frame

        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(
            config, [text]
        )
        timings, char_probs, state_list = ctc_segmentation.ctc_segmentation(
            config, log_probs[0].numpy(), ground_truth_mat
        )
        segments = ctc_segmentation.determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, [text]
        )

        # Construir resultado por palabra
        result = []
        word_timings = segments[0] if segments else []
        
        for i, word in enumerate(words):
            if i < len(word_timings):
                start, end, score = word_timings[i]
                result.append({
                    "word": word,
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "score": round(float(score), 4)
                })
            else:
                result.append({
                    "word": word,
                    "start": 0,
                    "end": 0,
                    "score": 0.0
                })

        return {"words": result}

    finally:
        os.unlink(tmp_path)
