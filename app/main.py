# app.py
import os
import datetime
import torch
import streamlit as st
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from audio_recorder_streamlit import audio_recorder
import PyPDF2
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
import asyncio
import time

# Set device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"

class RAGSystem:
    def __init__(self):
        self.vector_store = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                cache_folder="/tmp/hf_cache",
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            st.error(f"Embeddings initialization error: {str(e)}")
            raise

    def process_pdf(self, pdf_file) -> None:
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()

            chunks = self.text_splitter.split_text(text)
            self.vector_store = FAISS.from_texts(
                chunks,
                self.embeddings
            )
            st.success("PDF processing completed")
        except Exception as e:
            st.error(f"PDF processing error: {str(e)}")
            raise

    def query(self, question: str, k: int = 3) -> List[str]:
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Please process a PDF first.")
        try:
            return self.vector_store.similarity_search(question, k=k)
        except Exception as e:
            st.error(f"Query execution error: {str(e)}")
            raise

@st.cache_resource
def load_whisper_model():
    processor = WhisperProcessor.from_pretrained("openai/whisper-base")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-base").to(device)
    model.config.forced_decoder_ids = None
    return processor, model

def transcribe(audio_path, processor, model):
    """Transcribe audio file to text"""
    import librosa
    audio, sr = librosa.load(audio_path, sr=16000)
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    input_features = input_features.to(device)
    
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]

def initialize_session_state():
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    if 'current_pdf_name' not in st.session_state:
        st.session_state.current_pdf_name = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'recording_started' not in st.session_state:
        st.session_state.recording_started = False
    if 'last_process_time' not in st.session_state:
        st.session_state.last_process_time = 0

def add_message(role: str, content: str):
    """Add a message and trigger screen update"""
    if content.strip():  # Skip empty messages
        st.session_state.messages.append({
            "role": role,
            "content": content
        })

def display_chat_messages():
    """Display chat history"""
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"🗣️ **You**: {content}")
        elif role == "assistant":
            st.markdown(f"🤖 **Assistant**: {content}")

async def display_rag_chunks(contexts):
    """Display chunks retrieved by RAG"""
    if not contexts:
        return

    try:
        chunks_text = "\n\n".join([
            f"**Related Section {i+1}:**\n{doc.page_content}"
            for i, doc in enumerate(contexts)
        ])
        
        placeholder = st.empty()
        placeholder.markdown(f"🤖 **Assistant**: Here are the relevant information found:\n\n{chunks_text}")
        
        add_message("assistant", f"Here are the relevant information found:\n\n{chunks_text}")
            
    except Exception as e:
        st.error(f"Error displaying RAG chunks: {str(e)}")

async def process_audio(audio_bytes, processor, model):
    """Process audio data"""
    try:
        # Save temporary audio file
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        audio_path = f"audio_{timestamp}.wav"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)

        # Transcribe audio
        transcribed_text = transcribe(audio_path, processor, model)
        if transcribed_text.strip():
            add_message("user", transcribed_text)
            
            # Generate RAG query response
            if st.session_state.pdf_processed:
                contexts = st.session_state.rag_system.query(transcribed_text)
                await display_rag_chunks(contexts)
            else:
                add_message("assistant", "Please upload a PDF file before querying for information.")

    except Exception as e:
        st.error(f"Error during processing: {str(e)}")
    
    finally:
        # Delete temporary file
        if os.path.exists(audio_path):
            os.remove(audio_path)

async def main():
    st.title("Voice RAG Chatbot")
    initialize_session_state()

    # PDF settings in sidebar
    with st.sidebar:
        st.header("PDF Settings")
        pdf_file = st.file_uploader("Upload PDF File", type="pdf")
        
        if pdf_file is not None and (not st.session_state.pdf_processed or 
                                    st.session_state.current_pdf_name != pdf_file.name):
            with st.spinner("Processing PDF..."):
                st.session_state.rag_system.process_pdf(pdf_file)
                st.session_state.pdf_processed = True
                st.session_state.current_pdf_name = pdf_file.name

    # Load Whisper model
    processor, model = load_whisper_model()

    # Main chat area
    chat_container = st.container()
    with chat_container:
        display_chat_messages()
        
        # Voice input section
        st.write("---")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.write("🎤 **Voice Input** (Auto-process after 1.5s silence)")
        with col2:
            if st.button(
                "Stop Recording" if st.session_state.recording_started else "Start Recording",
                type="secondary" if st.session_state.recording_started else "primary",
                key="recording_button"
            ):
                st.session_state.recording_started = not st.session_state.recording_started
                st.rerun()

        # Show voice input only when recording is active
        if st.session_state.recording_started:
            audio_bytes = audio_recorder(
                pause_threshold=1.5,
                recording_color="#e74c3c",
                neutral_color="#666666",
                key="audio_recorder"
            )

            # Process new audio data if threshold time has passed
            current_time = time.time()
            if audio_bytes and (current_time - st.session_state.last_process_time) >= 1.5:
                st.session_state.last_process_time = current_time
                await process_audio(audio_bytes, processor, model)

if __name__ == "__main__":
    asyncio.run(main())