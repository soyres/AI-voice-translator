import uuid
import os
import gradio as gr
import assemblyai as aai
from translate import Translator
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs
from pathlib import Path

# get api keys
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ASSEEMBLYAI_API_KEY = os.getenv("ASSEEMBLYAI_API_KEY")


# main function that calls audio transcription, text translation and text to speech functions
def voice_to_voice(audio_file):
    
    #transcribe audio file
    transcription_response = audio_transcription(audio_file)

    # error handling
    if transcription_response.status == aai.TranscriptStatus.error:
        raise gr.Error(transcription_response.error)
    else:
        # text in english
        text = transcription_response.text
        print(f"text is {text}")

    es_translation = text_translation(text)
    
    es_audio_path = text_to_speech(es_translation)
    es_path = Path(es_audio_path)
    return es_path

    
# audio transcription function
def audio_transcription(audio_file):
    # call assemply AI to do transcription
    aai.settings.api_key = "e4c2ae0881304232a7ea3bbc97914cf0"
    transcriber = aai.Transcriber()

    transcription =  transcriber.transcribe(audio_file)

    return transcription

# translate text to specific language
def text_translation(text):

    translator_es = Translator(from_lang="en", to_lang="es")
    es_text = translator_es.translate(text)
    return es_text



# translates text to speech function
def text_to_speech(text: str):
    client = ElevenLabs(
        api_key=ELEVENLABS_API_KEY,
    )
   # Calling the text_to_speech conversion API with detailed parameters to create audio (voice)
    response = client.text_to_speech.convert(
        voice_id="nUTI3qORl8WhB5drXjX6", # custom voice
        optimize_streaming_latency="0",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_multilingual_v2", # use the turbo model for low latency, for other languages use the `eleven_multilingual_v2`
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
        ),
    )

    # Generating a unique file name for the output MP3 file
    save_file_path = f"{uuid.uuid4()}.mp3"

    # Writing the audio to a file
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")

    # Return the path of the saved audio file
    return save_file_path

# audio component
audio_input = gr.Audio(
    sources=["microphone"],
    type="filepath"
)

demo = gr.Interface(
    fn=voice_to_voice,
    inputs=audio_input,
    outputs=[gr.Audio(label="Spanish")]
)

if __name__ == "__main__":
    demo.launch()