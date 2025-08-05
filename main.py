import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
from predict import analizar_video  # Asegúrate que este archivo existe y tiene la función analizar_video

app = FastAPI()

# CORS para permitir peticiones desde Flutter (ajusta si usas IP o localhost diferente)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠️ Usa solo en desarrollo
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carpeta temporal para guardar videos
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.post("/scan")
async def scan_video(video: UploadFile = File(...)):
    try:
        # Guardar el archivo en el servidor
        video_path = os.path.join(UPLOAD_FOLDER, video.filename)
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(video.file, buffer)

        print(f"✅ Video recibido y guardado en: {video_path}")

        # Procesar el video con tu modelo
        resultados = analizar_video(video_path)

        print(f"📊 Resultados del modelo: {resultados}")

        return {
            "success": True,
            "resultado": resultados
        }

    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        return {
            "success": False,
            "error": str(e)
        }

    finally:
        # Eliminar el video después de procesarlo
        if os.path.exists(video_path):
            os.remove(video_path)

@app.get("/")
def home():
    return {"message": "Servidor de análisis de video funcionando"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
