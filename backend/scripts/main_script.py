# pip install torch-2.8.0+cu128-cp310-cp310-win_amd64.whl
# pip install llama-index==0.14.12
# pip install llama-index-core==0.14.12
# pip install llama-index-llms-huggingface==0.6.1
# pip install llama-index-embeddings-huggingface==0.6.1
# pip install llama_index.llms.llama_cpp==0.5.1
# pip install llama-index-vector-stores-chroma==0.5.5
# pip install transformers==4.57.6
# pip install sentence-transformers==5.2.0
# pip install accelerate==1.12.0
# pip install docx2txt==0.9
# pip install natasha==1.6.0
# pip install hf_xet==1.2.0
# pip install cmake==4.2.1


import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from llama_index.core import Settings, Document, VectorStoreIndex, SimpleDirectoryReader, StorageContext, PromptTemplate
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from natasha import Segmenter, Doc
import time
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROJECT_ROOT, RAW_DATA, MODELS

# --- 1. Загрузка документов ---

def preprocess_text(text):
    doc = Doc(text)
    segmenter = Segmenter()
    doc.segment(segmenter)
    preprocessed_text = ' '.join([token.text for token in doc.tokens])
    preprocessed_text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', preprocessed_text)
    return preprocessed_text

raw_path = RAW_DATA
documents = SimpleDirectoryReader(raw_path).load_data()
processed_documents = []
for doc in documents:
    processed_doc = Document(
        text=preprocess_text(doc.text), 
        metadata=doc.metadata)
    processed_documents.append(processed_doc)
    
print(f"Загружено {len(documents)} документ(ов).")

# --- 2. Настройка моделей через HuggingFace ---

# ВАРИАНТ 1: Локальная модель для генерации + эмбеддинги

Settings.embed_model = HuggingFaceEmbedding(
    model_name="intfloat/multilingual-e5-base"
)

status = torch.cuda.is_available()
print(('CUDA неактивна!', 'CUDA активна')[status])

# --- 2. Настройка моделей через LlamaCPP и эмбеддинги ---

# 2. Путь к вашему скачанному GGUF файлу
model_path = f"{MODELS}/Vikhr-7B-instruct_0.2.Q2_K.gguf"

# 3. Инициализация LLM через LlamaCPP (интеграция llama-cpp-python)
Settings.llm = LlamaCPP(
    model_path=model_path,
    
    # Ключевые параметры для управления памятью и производительностью
    model_kwargs={
        "n_gpu_layers": 20,  # Количество слоёв, загружаемых на GPU
        "n_ctx": 8192,       # Размер контекста
        "n_threads": 8,      # Количество CPU потоков
        "n_batch": 512,      # Размер батча для обработки
    },
    
    # Параметры генерации
    temperature=0.1,
    max_new_tokens=512,
    
    verbose=False
)

print(f"Модель {model_path} загружена через llama.cpp.")

# --- 3. Создание векторной базы данных ---
chroma_path = os.path.join(PROJECT_ROOT, "chroma_db")
chroma_client = chromadb.PersistentClient(path=chroma_path)
collectionname = "rag_assistant"
if any(collection.name == collectionname for collection in chroma_client.list_collections()):
    print('deleting collection')
    chroma_client.delete_collection(name=collectionname)
chroma_collection = chroma_client.get_or_create_collection(collectionname)
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# --- 4. Индексирование ---
index = VectorStoreIndex.from_documents(
    processed_documents,
    storage_context=storage_context,
    show_progress=True
)
print("Индексация завершена.")

# ... (ваш существующий код до создания query_engine остается без изменений) ...

# --- 5. Создание query_engine и запуск интерактивного режима ---
qa_prompt_str = """
Ты — помощник, который отвечает ТОЛЬКО на основе предоставленного контекста.
Если в контексте нет информации для ответа на вопрос, скажи: "На основе предоставленных документов я не могу ответить на этот вопрос."

Контекстная информация:
{context_str}

Вопрос: {query_str}

Требования:
1. Отвечай ТОЛЬКО на основе контекста выше
2. Не добавляй информацию, которой нет в контексте
3. Если нужно, цитируй конкретные части контекста
4. Максимально возможно старайся добавить пункты документа, в котором ты нашёл информацию
5. Если ответа нет в контексте, так и скажи

Ответ:"""

qa_prompt = PromptTemplate(qa_prompt_str)

# Создаем query_engine для использования в диалоге
query_engine = index.as_query_engine(
    similarity_top_k=3,
    text_qa_template=qa_prompt,
    response_mode="compact"  # или "refine" для более сложных ответов
)

def interactive_dialog():
    """Функция для интерактивного диалога в командной строке"""
    print("\n" + "="*60)
    print("Ассистент готов к работе!")
    print("Введите ваш вопрос или команду 'выход', 'exit', 'quit' - для завершения работы")
    print("="*60 + "\n")
    
    while True:
        try:
            # Получаем ввод пользователя
            user_input = input("\nВопрос: ").strip()
            
            # Проверяем команды для выхода
            if user_input.lower() in ['выход', 'exit', 'quit', ]:
                print("Завершение работы")
                break
            
            # Проверяем на пустой ввод
            elif not user_input:
                print("Пожалуйста, введите вопрос или команду.")
                continue
            
            # Обрабатываем обычный вопрос
            print("\nОбрабатываю запрос...")
            start = time.time()
            
            # Получаем ответ
            response = query_engine.query(user_input)
            
            # Выводим ответ
            print("\n" + "="*60)
            print(response)
            print("="*60)
            
            # Выводим время выполнения
            stop = time.time()
            total_time = divmod(round(stop - start), 60)
            print(f"Время на ответ: {total_time[0]}м. {total_time[1]}c.")
            
        except KeyboardInterrupt:
            # Обработка Ctrl+C
            print("\n\nПрервано пользователем. Для выхода введите 'выход' или 'exit'.")
            continue
        except EOFError:
            # Обработка Ctrl+D (конец файла)
            print("\n\nЗавершение работы.")
            break
        except Exception as e:
            # Обработка других ошибок
            print(f"\nПроизошла ошибка: {e}")
            print("Попробуйте снова или введите 'выход' для завершения.")

# Запускаем интерактивный диалог
if __name__ == "__main__":
    interactive_dialog()