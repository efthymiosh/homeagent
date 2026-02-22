from kokoro import KPipeline
import sounddevice as sd

pipeline = KPipeline(lang_code="a")
text = """
Hello, sir. This is Maude. How may I assist you today?
"""
generator = pipeline(text, voice="af_bella")
for i, (gs, ps, audio) in enumerate(generator):
    print(i, gs, ps)
    sd.play(audio, 24000)
    sd.wait()
