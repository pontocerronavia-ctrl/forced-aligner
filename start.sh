#!/bin/bash
if [ ! -d "models/vosk-model-small-es-0.42" ]; then
  echo "Descargando modelo Vosk..."
  wget -q https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip -O /tmp/vosk.zip
  unzip -q /tmp/vosk.zip -d models
  rm /tmp/vosk.zip
fi
uvicorn app:app --host 0.0.0.0 --port $PORT --workers 1
```

**2 — En Render**, cambiar el Start Command a:
```
bash start.sh
