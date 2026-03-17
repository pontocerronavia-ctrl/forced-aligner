from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from vosk import Model, KaldiRecognizer
import json
import wave
import tempfile
import os
import re
from rapidfuzz import process, fuzz

# Config: ruta del modelo
MODEL_PATH = os.environ.get("VOSK_MODEL_PATH", "models/vosk-model-small-es-0.42")

app = FastAPI(title="Vosk Forced-Align (light)")

# CORS — siempre retornar headers incluso en errores
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

@app.options("/{path:path}")
async def options_handler():
    return JSONResponse(
        content={},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "*",
        }
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo al inicio (fallará si no está presente)
try:
    model = Model(MODEL_PATH)
except Exception as e:
    model = None
    print(f"[WARN] Vosk model not loaded. Put model under {MODEL_PATH}. Error: {e}")


def ensure_wav_16k_mono(path: str):
    """Verifica que el WAV sea 16kHz mono; lanza Exception si no."""
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        rate = wf.getframerate()
        sampwidth = wf.getsampwidth()
    if channels != 1 or rate != 16000:
        raise ValueError(f"WAV must be mono (1) and 16kHz. Found channels={channels}, rate={rate}, sampwidth={sampwidth}")


def transcribe_with_vosk(wav_path: str):
    """Transcribe WAV (assumes 16k mono) y devuelve lista de words con start,end,conf."""
    wf = wave.open(wav_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)
    results = []
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            r = json.loads(rec.Result())
            if "result" in r:
                results.extend(r["result"])
    # Final
    r = json.loads(rec.FinalResult())
    if "result" in r:
        results.extend(r["result"])
    # Each item: { "conf": float, "start": float, "end": float, "word": str }
    return results


def tokenize_text(text: str):
    """Tokeniza texto en palabras unicode (minimiza signos de puntuación)."""
    # mantén acentos y caracteres unicode; devuelve lista en minúsculas
    tokens = re.findall(r"\b\w[\w'’-]*\b", text.lower(), flags=re.UNICODE)
    return tokens

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/align")
async def align(audio: UploadFile = File(...), text: str = Form(...)):
    """
    Recibe: form field 'text' (string) y file 'audio' (WAV 16kHz mono).
    Retorna: JSON { "aligned": [ {word, start, end, score}, ... ] }
    """
    if model is None:
        raise HTTPException(status_code=503, detail=f"Vosk model not loaded. Place model under {MODEL_PATH}")

    # Guardar wav temporalmente (asegura que el cliente suba WAV 16k mono)
    suffix = os.path.splitext(audio.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp_path = tmp.name
        tmp.write(await audio.read())
        tmp.flush()

    try:
        # Validación simple: exigir WAV 16k mono para evitar dependencia ffmpeg
        try:
            ensure_wav_16k_mono(tmp_path)
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Audio must be WAV 16kHz mono. Convert locally with ffmpeg: ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav. Error: {e}"
            )

        vosk_words = transcribe_with_vosk(tmp_path)  # list of dicts
        # Normalizar lista de palabras lower
        vosk_word_list = [w["word"].lower() for w in vosk_words]

        tokens = tokenize_text(text)
        aligned = []

        # Matching secuencial con ventana (evita empates repetidos)
        last_idx = 0
        window = 15  # lookahead window; configurable para mejores mapeos
        threshold = 60  # umbral de similitud (0-100)

        for token in tokens:
            # slice window from last_idx
            slice_end = min(last_idx + window, len(vosk_word_list))
            best = None
            if last_idx < len(vosk_word_list):
                sub = vosk_word_list[last_idx:slice_end]
                if sub:
                    m = process.extractOne(token, sub, scorer=fuzz.ratio)
                    if m:
                        match_str, score, rel_idx = m  # rel_idx es índice en sub
                        abs_idx = last_idx + rel_idx
                        if score >= threshold:
                            vw = vosk_words[abs_idx]
                            conf = vw.get("conf", score / 100.0)
                            aligned.append({
                                "word": token,
                                "start": float(vw["start"]),
                                "end": float(vw["end"]),
                                "score": float(conf)
                            })
                            last_idx = abs_idx + 1
                            continue  # siguiente token

            # fallback: buscar globalmente a partir de last_idx
            if last_idx < len(vosk_word_list):
                m = process.extractOne(token, vosk_word_list[last_idx:], scorer=fuzz.ratio)
                if m:
                    match_str, score, rel_idx = m
                    abs_idx = last_idx + rel_idx
                    if score >= (threshold - 10):  # umbral más laxo para fallback
                        vw = vosk_words[abs_idx]
                        conf = vw.get("conf", score / 100.0)
                        aligned.append({
                            "word": token,
                            "start": float(vw["start"]),
                            "end": float(vw["end"]),
                            "score": float(conf)
                        })
                        last_idx = abs_idx + 1
                        continue

            # Si no se encontró correspondencia, devolver null timestamps
            aligned.append({
                "word": token,
                "start": None,
                "end": None,
                "score": 0.0
            })

        return JSONResponse({"aligned": aligned})

    finally:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
