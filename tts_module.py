import queue
import threading
import logging
import time
import re
import os
import numpy as np
import sounddevice as sd
from piper import PiperVoice



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TTSModule")


# piper tts model parameters

TTS_MODEL_FILE: str = os.getenv("TTS_MODEL_FILE")
TTS_CONFIG_FILE: str = os.getenv("TTS_CONFIG_FILE")


# piper streaming tts module
# handles low-latency text-to-speech synthesis using piper ONNX models

class PiperStreamingTTS:

    # initializes the TTS voice model and starts background synthesis/playback threads
    def __init__(self, model_path: str | None = None, config_path: str | None = None) -> None:

        model_path = model_path or TTS_MODEL_FILE
        config_path = config_path or TTS_CONFIG_FILE

        if not model_path:
            raise ValueError("TTS_MODEL_FILE is not set in environment (.env) nor provided as argument.")

        logger.info("Loading TTS model from: %s (config: %s)...", model_path, config_path)
        try:
            self.voice = PiperVoice.load(model_path, config_path)
            logger.info("TTS model loaded successfully.")
        except Exception as e:
            logger.error("Failed to load Piper model: %s", e)
            raise

        self.audio_queue = queue.Queue()
        self.text_queue = queue.Queue()
        self.is_running = True
        
        # audio output parameters from voice config
        self.sample_rate = self.voice.config.sample_rate
        
        # worker threads for parallel synthesis and audio playback
        self.synthesis_thread = threading.Thread(target=self._synthesis_worker, daemon=True)
        self.playback_thread = threading.Thread(target=self._playback_worker, daemon=True)
        
        self.synthesis_thread.start()
        self.playback_thread.start()

    # clears pending text and audio queues
    def clear_queue(self) -> None:

        while not self.text_queue.empty():
            try:
                self.text_queue.get_nowait()
                self.text_queue.task_done()
            except queue.Empty:
                break
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
                self.audio_queue.task_done()
            except queue.Empty:
                break

    # adds a text chunk to the queue for immediate synthesis
    def process_chunk(self, text_chunk: str, risk_level: str = "default", on_first_audio: callable = None) -> None:

        if text_chunk and text_chunk.strip():
            # if high risk, clear pending queues to prioritize emergency warning
            if risk_level == "high":
                logger.info("High risk detected: canceling pending text/audio to prioritize emergency warning...")
                self.clear_queue()

            self.text_queue.put({
                "text": text_chunk.strip(),
                "risk": risk_level,
                "on_first_audio": on_first_audio,
            })

    # splits a full text by punctuation marks and queues each chunk for streaming
    def process_text_stream(self, text: str, risk_level: str = "default", on_first_audio: callable = None) -> None:

        if not text:
            return
        chunks = re.split(r'(?<=[.!?;,])\s+', text.strip())
        for idx, chunk in enumerate(chunks):
            if chunk.strip():
                # only attach on_first_audio callback to the very first chunk of the stream
                cb = on_first_audio if idx == 0 else None
                self.process_chunk(chunk.strip(), risk_level=risk_level, on_first_audio=cb)

    # waits until all queued text and audio chunks have been synthesized and played
    def wait_until_done(self, timeout: float = 3.0) -> None:
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < timeout:
            if self.text_queue.unfinished_tasks == 0 and self.audio_queue.unfinished_tasks == 0:
                break
            time.sleep(0.02)

    # synthesizes text with automatic fallback if length_scale is not supported by piper version
    def _synthesize_text(self, text: str, speed_scale: float):
        kwargs = {"length_scale": speed_scale}
        if hasattr(self.voice, "synthesize_stream_raw"):
            try:
                return self.voice.synthesize_stream_raw(text, **kwargs)  # type: ignore
            except TypeError:
                return self.voice.synthesize_stream_raw(text)
        elif hasattr(self.voice, "synthesize"):
            try:
                return self.voice.synthesize(text, **kwargs)  # type: ignore
            except TypeError:
                return self.voice.synthesize(text)
        raise AttributeError("PiperVoice instance does not support recognized synthesis methods.")


    # background worker thread that synthesizes text chunks into PCM audio
    def _synthesis_worker(self) -> None:


        while self.is_running:
            try:
                item = self.text_queue.get(timeout=1.0)
                if item is None: 
                    break # stop signal
                
                text = item["text"]
                risk = item["risk"]
                on_first_audio_cb = item.get("on_first_audio")
                
                # speed adjustment based on risk level
                speed_scale = 1.0 
                if risk == "high":
                    speed_scale = 0.8  # speak 20% faster (urgency)
                elif risk == "medium":
                    speed_scale = 0.9  # speak 10% faster
                
                logger.debug("[Synthesize] Risk: %s | Speed: %.1f | Text: %s", risk, speed_scale, text)
                
                try:
                    audio_stream = self._synthesize_text(text, speed_scale)
                except Exception as e:
                    logger.error("Audio synthesis error for text '%s': %s", text, e)
                    self.text_queue.task_done()
                    continue

                
                first_chunk = True
                for audio_bytes in audio_stream:
                    if first_chunk and on_first_audio_cb:
                        try:
                            on_first_audio_cb(time.perf_counter())
                        except Exception as cb_err:
                            logger.error("Error in on_first_audio callback: %s", cb_err)
                        first_chunk = False

                    if isinstance(audio_bytes, bytes):
                        audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                    elif isinstance(audio_bytes, np.ndarray):
                        audio_data = audio_bytes
                    else:
                        audio_data = np.frombuffer(bytes(audio_bytes), dtype=np.int16)
                    self.audio_queue.put(audio_data)

                self.text_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("Audio synthesis error: %s", e)


    # background worker thread that plays audio on the sound device
    def _playback_worker(self) -> None:

        stream = None
        try:
            stream = sd.OutputStream(samplerate=self.sample_rate, channels=1, dtype='int16')
            stream.start()
        except Exception as e:
            logger.warning("Could not open audio output device (%s). Audio will be processed without physical playback.", e)

        while self.is_running:
            try:
                audio_data = self.audio_queue.get(timeout=0.1)
                if audio_data is None:
                    break
                
                if stream is not None:
                    stream.write(audio_data)
                self.audio_queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                logger.error("Audio playback error: %s", e)
                
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as e:
                logger.error("Error closing audio stream: %s", e)

    # stops background synthesis and playback threads gracefully
    def shutdown(self) -> None:

        if not self.is_running:
            return
        self.is_running = False
        self.text_queue.put(None)
        self.audio_queue.put(None)
        if self.synthesis_thread.is_alive():
            self.synthesis_thread.join(timeout=2.0)
        if self.playback_thread.is_alive():
            self.playback_thread.join(timeout=2.0)
        logger.info("TTS module shut down.")


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


# test
if __name__ == "__main__":
    if not TTS_MODEL_FILE or not os.path.exists(TTS_MODEL_FILE):
        print("ATTENTION: Define 'TTS_MODEL_FILE' and 'TTS_CONFIG_FILE' in .env and ensure files exist for testing.")
        print("Example download links:")
        print("  - https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_PT/tugao/medium/pt_PT-tugao-medium.onnx")
        print("  - https://huggingface.co/rhasspy/piper-voices/resolve/main/pt/pt_PT/tugao/medium/pt_PT-tugao-medium.onnx.json")
    else:
        tts = PiperStreamingTTS()
        
        print("\n--- Streaming TTS Test ---")
        
        print("Sending chunk 1...")
        tts.process_chunk("Atenção.", risk_level="high")
        time.sleep(0.5)
        
        print("Sending chunk 2...")
        tts.process_chunk("Automóvel a bloquear a passadeira.", risk_level="high")
        time.sleep(1)
        
        print("Sending chunk 3 (Low risk)...")
        tts.process_chunk("Passeio livre à sua direita.", risk_level="low")
        
        time.sleep(5)
        tts.shutdown()

