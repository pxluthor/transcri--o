#from reportlab.lib.units import cm
#from reportlab.lib.pagesizes import A4
#from reportlab.pdfgen import canvas
#from pathlib import Path


import os
import streamlit as st
from pydub import AudioSegment
from groq import Groq
import google.generativeai as genai
from docx import Document
from fpdf import FPDF
import toml

from pydub import AudioSegment
from pydub.utils import which


# Instala ffmpeg se não estiver presente
os.system("apt-get update && apt-get install -y ffmpeg")

# Certifica de que o pydub encontra ffmpeg e ffprobe
AudioSegment.converter = which("ffmpeg")
AudioSegment.ffmpeg = which("ffmpeg")
AudioSegment.ffprobe = which("ffprobe")

#chaves API
api_key_groq = st.secrets["api_keys"]["api_key2"]
api_key_gemini = st.secrets["api_keys"]["api_key1"]

# Configuração da API Groq whisper

client = Groq(api_key=api_key_groq)
model = 'whisper-large-v3'

# Configuração da API Gemini

genai.configure(api_key=api_key_gemini)
generation_config = {
    "temperature": 0.5,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model_g = genai.GenerativeModel(model_name='models/gemini-1.5-flash-latest', generation_config=generation_config)

# Função para dividir o áudio
def split_audio(filepath, chunk_length_ms=180000): # 3 minutos
    audio = AudioSegment.from_file(filepath)  # segmento de audio
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    return chunks

# converte chunk para texto.
def audio_chunk_to_text(chunk, model, client):
    temp_filename = "temp_chunk.mp3"
    chunk.export(temp_filename, format="mp3")

    with open(temp_filename, "rb") as file:
        translation = client.audio.transcriptions.create(
            file=(temp_filename, file.read()),
            model=model,
            prompt="realize a transcrição retornando também o tempo",
            language="pt"
        )

    os.remove(temp_filename)
    return translation.text

#função que transcreve o audio
def transcribe_audio(filepath, model, client):
    chunks = split_audio(filepath)
    full_transcription = []
    status_text = st.empty()
    for i, chunk in enumerate(chunks):
        status_text.text(f"Transcrevendo parte {i + 1} de {len(chunks)}...")
        text = audio_chunk_to_text(chunk, model, client)
        full_transcription.append(text)
        status_text.text("Transcrição completa!")
    return full_transcription


# função exportar PDF
def export_to_pdf(transcription):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for line in transcription:
        line_str = line.encode('latin1', 'ignore').decode('latin1') 
        pdf.multi_cell(0, 10, line_str)
    pdf_file = "transcription.pdf"
    pdf.output(pdf_file)
    return pdf_file


# função exportar PDF
def export_to_docx(transcription):
    doc = Document()
    for line in transcription:
        doc.add_paragraph(line)
    doc_file = "transcription.docx"
    doc.save(doc_file)
    return doc_file


def role_to_streamlit(role):
    return "assistente" if role == "model" else role

def main():
    st.title("💬 Chat - Transcription audio 🎙🔉")

    with st.sidebar:
        st.button("NOVO CHAT", on_click=limpar_chat)

        # Mostrar os botões de download apenas se a transcrição estiver pronta
        if "transcricao_feita" in st.session_state and st.session_state.transcricao_feita:
            if "pdf_downloads" in st.session_state:
                with open("transcription.pdf", "rb") as f:
                    st.download_button(
                        label=f"Download PDF ({st.session_state.pdf_downloads})",
                        data=f,
                        file_name="transcription.pdf",
                        mime="application/pdf",
                        on_click=lambda: st.session_state.pdf_downloads + 1
                    )
            if "docx_downloads" in st.session_state:
                with open("transcription.docx", "rb") as f:
                    st.download_button(
                        label=f"Download DOCX ({st.session_state.docx_downloads})",
                        data=f,
                        file_name="transcription.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        on_click=lambda: st.session_state.docx_downloads + 1
                    )

    if "chat" not in st.session_state:
        st.session_state.chat = []
        st.session_state.history = []

  

    for message in st.session_state.chat:
        role = "user" if message["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message['text'])        

    opcao_entrada = st.sidebar.radio("Selecione o tipo de entrada:", ("Texto", "Áudio"))
    
    if opcao_entrada == "Texto":
        
        if prompt := st.chat_input("Como posso ajudar?"): 
            st.session_state.chat.append({"role": "user", "text": prompt})
            st.session_state.history.append({"role": "user", "content": prompt})

            # exibe a mensagem do user
            with st.chat_message("user"):
                st.markdown(prompt)

            #processamento da pergunta do usuário
            response = client.chat.completions.create(
                messages=st.session_state.history,
                model="llama3-8b-8192"
            )
            response_text = response.choices[0].message.content # resposta da IA

            #Adiciona as mensagens no chat e no histórico
            st.session_state.chat.append({"role": "assistente", "text": response_text})
            st.session_state.history.append({"role": "assistant", "content": response_text})

            # exibe a resposta da IA
            with st.chat_message("assistente"):
                st.markdown(response_text)
                
    else:
        #botão upload
        arquivo_carregado = st.file_uploader("Carregar arquivo de áudio (MP3)")

        if arquivo_carregado:
            st.sidebar.markdown("# PLAY AUDIO 🔉 ")

            # carregar o arquivo
            @st.cache_data
            def carregar_audio(arquivo_carregado):
                return arquivo_carregado.read()

            audio_data = carregar_audio(arquivo_carregado)
            with open("audio_temp.mp3", "wb") as f:
                f.write(audio_data)

            st.sidebar.audio("audio_temp.mp3", format="audio/mpeg", loop=False)
            st.sidebar.info("Audio carregado !")


            # verifica na sessão se a transcrição foi feita
            if "transcricao_feita" not in st.session_state:
                st.session_state.transcricao_feita = False
            if "transcricao" not in st.session_state:
                st.session_state.transcricao = ""

            if not st.session_state.transcricao_feita and st.button("Fazer transcrição"):
                st.session_state.file_path = "audio_temp.mp3"
                transcription = transcribe_audio(st.session_state.file_path, model, client)
                formatted_transcription = transcription

                prompt2 = f'''Responda sempre em português do Brasil. 
                            Você trabalha na Leste telecom, o seu trabalho é realizar a transcrição de conversas identificando, transcrevendo e fazendo correções 
                            de algumas palavras dentro do contexto da fala de cada interlocutor. 
                            Revise a conversa de acordo com o contexto:{formatted_transcription}. Retorne também o tempo correto de cada fala. como no exemplo:

                            Modelo:
                            tempo de fala -
                            interlocutor1: (fala do interlocutor)(\n).
                            tempo de fala -
                            interlocutor2: (fala do interlocutor)(\n).
                            Retorne a resposta da transcrição conforme o modelo apresentado. 
                            Use quebras de linha se necesário.

                            contexto:
                            O cliente entra em contato por telefone com a central da Leste telecom, geralmente quem inicia a conversa é o atendente que faz a saudação e 
                            pergunta como pode ajudar. 
                            '''
                
                resp = model_g.generate_content(prompt2)
                response_final = resp.text

                with st.chat_message("assistente"):
                    st.markdown(response_final)
                    st.session_state.chat.append({"role": "assistente", "text": response_final})

                # Botões de exportação
                pdf_file = export_to_pdf(response_final.splitlines())
                with open(pdf_file, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name=pdf_file,
                        mime="application/pdf"
                    )

                docx_file = export_to_docx(response_final.splitlines())
                with open(docx_file, "rb") as f:
                    st.download_button(
                        label="Download DOCX",
                        data=f,
                        file_name=docx_file,
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )    

                st.session_state.transcricao_feita = True
                st.session_state.transcricao = response_final
                st.session_state.pdf_downloads = 0
                st.session_state.docx_downloads = 0
                
              
             

                pdf_file = export_to_pdf(response_final.splitlines())
                docx_file = export_to_docx(response_final.splitlines())

            # perguntas sobre a transcrição.
            if st.session_state.transcricao_feita:
                if prompt3 := st.chat_input("Como posso ajudar?"): 
                    st.session_state.chat.append({"role": "user", "text": prompt3})
                    response = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt3},
                                  {"role": "system", "content": st.session_state.transcricao}],
                        model="llama3-70b-8192"
                    )
                    response_text = response.choices[0].message.content
                    with st.chat_message("assistente"):
                        st.markdown(response_text)
                        st.session_state.chat.append({"role": "assistente", "text": response_text})

def limpar_chat():
    st.session_state.chat = []
    st.session_state.history = []
    st.session_state.transcricao_feita = False
    if os.path.exists("audio_temp.mp3"):
        os.remove("audio_temp.mp3")

if __name__ == "__main__":
    main()