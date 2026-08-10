# PICO audio/video sincronizado con HandUMI

Esta extensión conserva el flujo normal de `handumi record`: los sensores USB/ZED,
Feetech y tracking PICO se muestrean y se guardan primero con LeRobot v3. El APK
HandUMI de XRoboToolkit graba en paralelo la cámara VST estéreo y el micrófono. Al
finalizar, HandUMI descarga cada toma por ADB, la alinea contra el reloj del PICO y
la incorpora al mismo dataset antes de validarlo o subirlo a Hugging Face.

## Pipeline

```text
Botón/voz de inicio
  -> device_control_json(CameraRecord, episode_id)
  -> APK abre VST + H.264 y micrófono + AAC
  -> manifest.inprogress.json cambia a "recording"
  -> HandUMI empieza a guardar filas LeRobot
  -> todas las filas incluyen observation.tracking.device_time_ns
  -> fin del episodio: APK cierra y publica manifest.json de forma transaccional
  -> adb pull de /sdcard/Download/HandUMI/<episode_id>
  -> ffmpeg divide side-by-side, remuestrea y recorta exactamente N frames
  -> faster-whisper produce segmentos y palabras con tiempos
  -> Parquet recibe índices/timestamps/errores/texto y metadata de medios
  -> validación LeRobot v3 -> push opcional a Hugging Face
```

El inicio tiene un *handshake*: HandUMI no toma el primer frame hasta comprobar por
ADB que cámara y micrófono están realmente activos. El final espera el manifiesto
completo y tres tamaños de video estables para no copiar un MP4 cuyo `moov` todavía
no se haya cerrado.

## Sincronización

XRoboToolkit ya escribe `device_time_ns` desde el PICO en cada muestra de tracking.
El APK usa el mismo reloj Unix epoch en nanosegundos al iniciar video y audio. Por
eso no se alinea con la hora del PC ni con la latencia USB.

- Video: se selecciona el segmento que corresponde al primer `device_time_ns`, se
  convierte a la frecuencia del dataset y se obliga a producir exactamente el
  número de filas del episodio. `sample_time_ns`, `sequence` y `healthy` quedan en
  cada fila. La grabación falla si menos de 95% queda dentro de `--max-sync-skew-s`.
- Audio: se conserva el M4A AAC mono de 48 kHz completo. Cada fila guarda si el
  audio está activo, el índice de muestra más cercano, su timestamp y el error de
  cuantización (normalmente menor de 10.5 microsegundos).
- Texto: `observation.language.pico_transcript` contiene el segmento hablado que
  cruza ese frame; `observation.language.pico_episode_transcript` repite el texto
  completo para consumo semántico. El JSONL conserva segmentos, palabras,
  probabilidad y tiempos para reconstruir otra granularidad sin retranscribir.

## Estructura añadida al dataset

```text
videos/observation.images.pico_head_left/chunk-000/file-000.mp4
videos/observation.images.pico_head_right/chunk-000/file-000.mp4
audio/pico_microphone/chunk-000/file-000.m4a
transcripts/pico_microphone/chunk-000/file-000.jsonl
.pico_av_raw/episode_000000_<id>/manifest.json
```

`meta/info.json` registra el esquema en `handumi.pico_av`. Los Parquet de episodios
incluyen las rutas de audio/transcript y los offsets; los Parquet de datos contienen
las columnas sincronizadas por frame. `.pico_av_raw` se conserva como evidencia para
reanálisis; se puede excluir manualmente de una copia si no se necesita auditoría.

## Requisitos y uso

1. PICO 4 Ultra Enterprise con acceso VST aprobado por PICO Enterprise.
2. APK modificado instalado, abierto y conectado al mismo PC Service.
3. Cable USB con depuración ADB autorizada; `adb devices` debe mostrar `device`.
4. Permisos Android `CAMERA` y `RECORD_AUDIO` concedidos.
5. `ffmpeg`/`ffprobe` y dependencias de `uv sync` instaladas.

Obtenga el SN mostrado por XRoboToolkit y ejecute, conservando los demás argumentos
de su grabación habitual:

```bash
uv sync
handumi record \
  --device pico \
  --pico-adb \
  --pico-av \
  --pico-device-id PICO_SN_AQUI \
  --pico-adb-serial ADB_SERIAL_AQUI \
  --pico-language es \
  --pico-transcription-model small
```

La primera transcripción descarga el modelo de faster-whisper. Use
`--no-pico-transcribe` sólo si desea guardar audio y campos de sincronización sin
texto. Por seguridad, `--resume` no está habilitado con PICO A/V: cada dataset nuevo
debe contener exactamente una toma PICO por episodio guardado.

## Fallos que invalidan un episodio

- No aparece el manifiesto `recording` en 15 segundos.
- La cámara VST o el micrófono no arrancan/permisos denegados.
- El MP4/M4A falta o está vacío.
- El conteo de filas no coincide con la captura.
- Más de 5% de frames excede el skew permitido.
- `ffmpeg`, transcripción o escritura de Parquet falla.

En esos casos el buffer LeRobot no se confirma y el dataset no se sube como válido.
