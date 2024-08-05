# código funcional trazendo a transcrição junto pela groq

import os
import streamlit as st
import toml
from pathlib import Path
import pyaudio
from st_audiorec import st_audiorec
import streamlit.components.v1 as components
from pydub import AudioSegment
from groq import Groq
import speech_recognition as sr

#config = toml.load("config.toml")

# Carregar a chave de API diretamente do arquivo de configuração
#api_key = st.secrets['api_keys']['groq'] 
#client = Groq(api_key=api_key)

api_key = "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"
client = Groq(api_key=api_key)
model = 'whisper-large-v3'


# função para dividir o audio 
def split_audio(filepath, chunk_length_ms=180000):
    """Dividir o arquivo de áudio em partes menores."""
    audio = AudioSegment.from_file(filepath)
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks
#transcrição
def audio_chunk_to_text(chunk, model, client):
    """Converter um pedaço de áudio em texto usando o modelo de transcrição."""
    temp_filename = "temp_chunk.mp3"
    chunk.export(temp_filename, format="mp3")
    with open(temp_filename, "rb") as file:
        translation = client.audio.transcriptions.create(
            file=(temp_filename, file.read()),
            model=model,
            
        )

    os.remove(temp_filename)  # Remove o arquivo temporário
    return translation.text

def transcribe_audio(filepath, model, client):
    """Transcrever todo o áudio dividindo-o em partes menores."""
    chunks = split_audio(filepath)
    full_transcription = ""
    for i, chunk in enumerate(chunks):
        st.write(f"Transcrevendo parte {i + 1} de {len(chunks)}...")
        text = audio_chunk_to_text(chunk, model, client)
        full_transcription += text + " \n"
    return full_transcription.strip()

# Função para converter o papel de parede
def role_to_streamlit(role):
    return "assistente" if role == "model" else role

# Função principal
def main():
    #components.iframe("https://typebot_view.pxluthor.com.br/l-e-s-t-e-linear-6ezuy34", height=600)
    st.title("💬 Chat - Transcription audio 🎙🔉")

    # Sidebar
    with st.sidebar:
        st.title("Gravação de Áudio")
        wav_audio_data = st_audiorec()
        st.divider()
        st.button("Limpar Chat", on_click=limpar_chat)

    # Inicialização do chat
    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Exibir histórico do chat
    for message in st.session_state.chat:
        with st.chat_message(role_to_streamlit(message['role'])):
            st.markdown(message['text'])

    # Opções de entrada: texto ou áudio
    opcao_entrada = st.sidebar.radio("Selecione o tipo de entrada:", ("Texto", "Áudio"))

    if opcao_entrada == "Texto":
        # ... Código para entrada de texto existente ...
        if prompt := st.chat_input("Como posso ajudar?",): 
            st.session_state.chat.append({"role": "user", "text": prompt})
            # Ajuste conforme a integração do modelo
            response = client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt}
                ],
                model="llama3-8b-8192"
            )
            response_text = response.choices[0].message.content
            with st.chat_message("assistente"):
                st.markdown(response_text) 
                st.session_state.chat.append({"role": "assistente", "text": response_text})

    else:
        # ... Código para entrada de áudio existente ...
        arquivo_carregado = st.file_uploader("Carregar arquivo de áudio (MP3 ou WAV)")
        st.chat_input("Como posso ajudar?")
        if arquivo_carregado:
            # Use st.cache_data para armazenar o arquivo em cache no diretório do Streamlit
            st.audio("audio_temp.mp3", format="audio/mpeg", loop=False)
            @st.cache_data
            def carregar_audio(arquivo_carregado):
                return arquivo_carregado.read()

            audio_data = carregar_audio(arquivo_carregado)
            with open("audio_temp.mp3", "wb") as f:
                f.write(audio_data)

            st.info("Audio carregado !")

            if st.button("Fazer transcrição"):
                # Use o caminho do arquivo em st.session_state
                st.session_state.file_path = "audio_temp.mp3"
                transcription = transcribe_audio(st.session_state.file_path, model, client)
                st.session_state.chat.append({"role": "user", "text": f"ajuste a transcrição intercalando os interlocutores na timeline, traga o horário da mensagem {transcription}"})
                with st.chat_message("assistente"):
                    st.success('Transcrição realizada')
                    st.markdown(transcription)

# Função para limpar o chat
def limpar_chat():
    st.session_state.chat = []
    if os.path.exists("audio_temp.mp3"):
        os.remove("audio_temp.mp3")

if __name__ == "__main__":
    main()
