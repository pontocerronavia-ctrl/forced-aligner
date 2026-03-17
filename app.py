from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchaudio
import tempfile
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/align")
async def align(
    audio: UploadFile = File(...),
    text: str = Form(...)
):
    from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
    import ctc_segmentation

    MODEL_ID = "facebook/wav2vec2-base-960h"
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.eval()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        tmp_path = tmp.name

    try:
        waveform, sr = torchaudio.load(tmp_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(dim=0)

        inputs = processor(
            waveform.numpy(),
            sampling_rate=16000,
            return_tensors="pt"
        )
        with torch.no_grad():
            logits = model(**inputs).logits

        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        vocab = processor.tokenizer.get_vocab()
        char_list = [k for k, v in sorted(vocab.items(), key=lambda x: x[1])]

        config = ctc_segmentation.CtcSegmentationParameters()
        config.char_list = char_list
        config.index_duration = logits.shape[1] / waveform.shape[0] * 16000
        config.index_duration = waveform.shape[0] / 16000 / logits.shape[1]

        ground_truth_mat, utt_begin_indices = ctc_segmentation.prepare_text(
            config, [text]
        )
        timings, char_probs, _ = ctc_segmentation.ctc_segmentation(
            config, log_probs[0].numpy(), ground_truth_mat
        )
        segments = ctc_segmentation.determine_utterance_segments(
            config, utt_begin_indices, char_probs, timings, [text]
        )

        words = text.strip().split()
        result = []
        for i, word in enumerate(words):
            if segments and i < len(segments):
                start, end, score = segments[i][0], segments[i][1], segments[i][2]
                result.append({
                    "word": word,
                    "start": round(float(start), 3),
                    "end": round(float(end), 3),
                    "score": round(float(score), 4)
                })
            else:
                result.append({
                    "word": word,
                    "start": 0.0,
                    "end": 0.0,
                    "score": 0.0
                })

        return {"words": result}

    finally:
        os.unlink(tmp_path)
