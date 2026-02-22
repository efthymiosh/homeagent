from RealtimeSTT import AudioToTextRecorder

def process_text(text):
    print (text)
    
if __name__ == "__main__":
    with AudioToTextRecorder(
        language="en",
        compute_type="int8",
        post_speech_silence_duration=0.2,
        enable_realtime_transcription=True,
        realtime_model_type="tiny.en",
        model="tiny.en",
        no_log_file=True,
    ) as recorder:
        while True:
            recorder.text(process_text)
