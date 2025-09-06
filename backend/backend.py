from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any, Generator
import sqlite3
import hashlib
import uuid
import os
import re
import smtplib
import PyPDF2
import docx
import io
import json
from datetime import datetime, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jose import JWTError, jwt
from passlib.context import CryptContext
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
import asyncio
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SECRET_KEY = "9cc1e4d7ded6927549a75b818dae083195882a2d985db1dfe258c3516e0b5496"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
EMAIL_HOST = "smtp.gmail.com"
EMAIL_PORT = 587
EMAIL_USER = "sujayc331@gmail.com"  # Your email
EMAIL_PASSWORD = "qial tpfo ddzx eomm"  # Your app password

app = FastAPI(title="LLM Chat API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Initialize ChromaDB with error handling
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    logger.info("ChromaDB client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize ChromaDB: {e}")
    chroma_client = None

# Initialize embedding model with error handling
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Embedding model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    embedding_model = None

# Database setup
def init_database():
    try:
        conn = sqlite3.connect('chat_app.db')
        c = conn.cursor()
        
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      email TEXT UNIQUE,
                      password TEXT,
                      full_name TEXT,
                      is_active BOOLEAN DEFAULT TRUE,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS conversations
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      title TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS messages
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      conversation_id INTEGER,
                      content TEXT,
                      is_user BOOLEAN,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      FOREIGN KEY (conversation_id) REFERENCES conversations (id))''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS documents
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      user_id INTEGER,
                      filename TEXT,
                      file_path TEXT,
                      file_type TEXT,
                      uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                      processed BOOLEAN DEFAULT FALSE,
                      FOREIGN KEY (user_id) REFERENCES users (id))''')
        
        c.execute('''CREATE TABLE IF NOT EXISTS email_templates
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      name TEXT UNIQUE,
                      subject TEXT,
                      body TEXT,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        
        # Insert default email templates
        c.execute('''INSERT OR IGNORE INTO email_templates (name, subject, body) 
                     VALUES (?, ?, ?)''', 
                  ('welcome', 'Welcome to LLM Chat App', 
                   'Hi {name},\n\nWelcome to our LLM Chat Application! We''re excited to have you on board.\n\nBest regards,\nThe Team'))
        
        c.execute('''INSERT OR IGNORE INTO email_templates (name, subject, body) 
                     VALUES (?, ?, ?)''', 
                  ('summary', 'Your Conversation Summary', 
                   'Hi {name},\n\nHere is a summary of your recent conversation:\n\n{summary}\n\nBest regards,\nThe Team'))
        
        conn.commit()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
    finally:
        conn.close()

init_database()

# Pydantic models
class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

class User(BaseModel):
    id: int
    email: str
    full_name: str
    is_active: bool

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class ChatMessage(BaseModel):
    content: str
    conversation_id: Optional[int] = None

class Conversation(BaseModel):
    id: int
    title: str

    class Config:
        from_attributes = True

class EmailRequest(BaseModel):
    template_name: str
    recipient_email: EmailStr
    parameters: Optional[dict] = {}

class DocumentUploadResponse(BaseModel):
    message: str
    document_id: int
    chunks_count: Optional[int] = None
    error: Optional[str] = None

# Database functions
def get_db_connection():
    return sqlite3.connect('chat_app.db')

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_user(email, password, full_name):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("INSERT INTO users (email, password, full_name) VALUES (?, ?, ?)",
                 (email, hash_password(password), full_name))
        conn.commit()
        
        # Send welcome email
        send_email('welcome', email, {'name': full_name})
        
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def get_user_by_email(email: str):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT id, email, password, full_name, is_active FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    
    if user:
        return {
            "id": user[0],
            "email": user[1],
            "password": user[2],
            "full_name": user[3],
            "is_active": user[4]
        }
    return None

def authenticate_user(email: str, password: str):
    user = get_user_by_email(email)
    if not user:
        return False
    if not verify_password(password, user["password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = get_user_by_email(email)
    if user is None:
        raise credentials_exception
    return user

# Email functions
def send_email(template_name: str, recipient_email: str, parameters: dict = {}):
    """Send email to the specified recipient using a template"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("SELECT subject, body FROM email_templates WHERE name = ?", (template_name,))
        template = c.fetchone()
        conn.close()
        
        if not template:
            logger.error(f"Email template '{template_name}' not found")
            return False
        
        subject = template[0]
        body = template[1]
        
        # Replace parameters in template
        for key, value in parameters.items():
            body = body.replace(f"{{{key}}}", str(value))
            subject = subject.replace(f"{{{key}}}", str(value))
        
        # Create email message
        msg = MIMEMultipart()
        msg['From'] = EMAIL_USER
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        # Send email
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.ehlo()
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.sendmail(EMAIL_USER, recipient_email, msg.as_string())
        server.quit()
        
        logger.info(f"Email sent successfully to {recipient_email}")
        return True
        
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP Authentication Error: Check your email credentials")
        return False
    except smtplib.SMTPException as e:
        logger.error(f"SMTP Error: {e}")
        return False
    except Exception as e:
        logger.error(f"Email error: {e}")
        return False

# Document processing functions
def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def extract_text_from_docx(file):
    doc = docx.Document(io.BytesIO(file))
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

def extract_text_from_csv(file):
    # Simple CSV extraction - could be enhanced with pandas
    return file.decode('utf-8')

def extract_text_from_json(file):
    data = json.loads(file.decode('utf-8'))
    # Convert JSON to readable text
    if isinstance(data, dict):
        return json.dumps(data, indent=2)
    return str(data)

def extract_text_from_file(file, filename, file_type):
    if file_type == 'pdf':
        return extract_text_from_pdf(io.BytesIO(file))
    elif file_type == 'docx':
        return extract_text_from_docx(file)
    elif file_type == 'csv':
        return extract_text_from_csv(file)
    elif file_type == 'json':
        return extract_text_from_json(file)
    elif file_type in ['txt', 'md']:
        return file.decode('utf-8')
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

# Advanced text processing
def advanced_text_chunking(text: str, file_type: str):
    """Advanced text chunking based on file type"""
    chunks = []
    
    if file_type == 'pdf':
        # For PDFs, try to preserve document structure
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs:
            if len(para.strip()) > 50:  # Only include meaningful paragraphs
                chunks.append(para.strip())
    
    elif file_type == 'docx':
        # For Word documents, preserve paragraph structure
        paragraphs = re.split(r'\n\s*\n', text)
        for para in paragraphs:
            if len(para.strip()) > 30:
                chunks.append(para.strip())
    
    elif file_type in ['csv', 'json']:
        # For structured data, create smaller chunks
        lines = text.split('\n')
        current_chunk = []
        for line in lines:
            if line.strip():
                current_chunk.append(line.strip())
                if len('\n'.join(current_chunk)) > 300:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
    
    else:
        # Default chunking for text files
        words = text.split()
        current_chunk = []
        for word in words:
            current_chunk.append(word)
            if len(' '.join(current_chunk)) > 400:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
        if current_chunk:
            chunks.append(' '.join(current_chunk))
    
    return chunks

# RAG functions
def get_collection(user_id: str):
    if chroma_client is None:
        raise HTTPException(status_code=500, detail="Vector database not available")
        
    collection_name = f"user_{user_id}_documents"
    try:
        collection = chroma_client.get_collection(collection_name)
    except:
        collection = chroma_client.create_collection(collection_name)
    return collection

def process_document(text: str, metadata: dict, user_id: str, file_type: str):
    collection = get_collection(user_id)
    
    # Advanced text chunking based on file type
    chunks = advanced_text_chunking(text, file_type)
    
    if embedding_model:
        embeddings = embedding_model.encode(chunks).tolist()
    else:
        # Fallback to simple hashing if embedding model is not available
        embeddings = [[hash(chunk) % 1000 / 1000 for _ in range(384)] for chunk in chunks]
    
    ids = [f"chunk_{i}_{uuid.uuid4().hex}" for i in range(len(chunks))]
    
    # Enhanced metadata with chunk information
    enhanced_metadata = []
    for i, chunk in enumerate(chunks):
        chunk_meta = metadata.copy()
        chunk_meta["chunk_id"] = i
        chunk_meta["chunk_count"] = len(chunks)
        enhanced_metadata.append(chunk_meta)
    
    collection.add(
        embeddings=embeddings,
        documents=chunks,
        ids=ids,
        metadatas=enhanced_metadata
    )
    
    return len(chunks)

def query_rag(query: str, user_id: str, top_k: int = 5):
    """Enhanced RAG query with better relevance scoring"""
    try:
        collection = get_collection(user_id)
        
        if embedding_model:
            query_embedding = embedding_model.encode([query]).tolist()[0]
        else:
            query_embedding = [hash(query) % 1000 / 1000 for _ in range(384)]
        
        # Get more results initially for better filtering
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2  # Get more results for filtering
        )
        
        # Simple relevance filtering
        relevant_docs = []
        for i, doc in enumerate(results['documents'][0]):
            # Basic relevance check - could be enhanced
            if any(keyword.lower() in doc.lower() for keyword in query.lower().split()[:3]):
                relevant_docs.append(doc)
            if len(relevant_docs) >= top_k:
                break
        
        return {"documents": [relevant_docs] if relevant_docs else results['documents']}
    except Exception as e:
        logger.error(f"RAG query error: {e}")
        return {"documents": [[]]}

def detect_topic_change(current_query: str, conversation_history: List[dict]) -> bool:
    """Detect if the user is changing topics"""
    if not conversation_history or len(conversation_history) < 3:
        return False
    
    # Get last few messages for context
    recent_messages = [msg["content"] for msg in conversation_history[-3:] if msg["is_user"]]
    
    # Simple topic change detection
    current_topics = set(current_query.lower().split()[:5])
    previous_topics = set()
    
    for msg in recent_messages:
        previous_topics.update(msg.lower().split()[:5])
    
    # If less than 20% overlap, likely topic change
    overlap = current_topics.intersection(previous_topics)
    return len(overlap) / len(current_topics) < 0.2 if current_topics else False

# Streaming response generator
async def generate_streaming_response(prompt: str, context: List[str], history: List[dict]):
    """Generate response with streaming"""
    context_str = "\n\n".join(context) if context else "No relevant context found."
    
    history_str = ""
    if history:
        for msg in history[-6:]:
            role = "User" if msg["is_user"] else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
    
    full_prompt = f"""Based on the following context and conversation history, answer the user's question.

Context:
{context_str}

Conversation History:
{history_str}

User: {prompt}

Assistant:"""
    
    try:
        # Use Ollama's streaming API
        stream = ollama.chat(
            model='llama2',
            messages=[{'role': 'user', 'content': full_prompt}],
            stream=True
        )
        
        for chunk in stream:
            if 'message' in chunk and 'content' in chunk['message']:
                yield chunk['message']['content']
                
    except Exception as e:
        yield f"Sorry, I encountered an error: {str(e)}"

def generate_response(query: str, context: List[str], conversation_history: List[dict] = None, topic_change: bool = False):
    context_str = "\n\n".join(context) if context else "No relevant context found."
    
    history_str = ""
    if conversation_history:
        for msg in conversation_history[-6:]:
            role = "User" if msg["is_user"] else "Assistant"
            history_str += f"{role}: {msg['content']}\n"
    
    # Enhanced prompt with topic awareness
    if topic_change and context:
        prompt = f"""I notice we're changing topics. Based on the user's documents, I found some relevant information.

Context from documents:
{context_str}

Conversation History:
{history_str}

User's new question: {query}

Please provide a helpful response that acknowledges the topic change and uses the document context when relevant:"""
    else:
        prompt = f"""Based on the following context and conversation history, answer the user's question.

Context from documents:
{context_str}

Conversation History:
{history_str}

User: {query}

Assistant:"""
    
    try:
        response = ollama.chat(model='llama2', messages=[{'role': 'user', 'content': prompt}])
        return response['message']['content']
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}. Please make sure Ollama is installed and running."

# API routes
@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["email"]}, expires_delta=access_token_expires
    )
    
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/register")
async def register_user(user: UserCreate):
    if get_user_by_email(user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    if create_user(user.email, user.password, user.full_name):
        return {"message": "User created successfully"}
    else:
        raise HTTPException(status_code=400, detail="Error creating user")

@app.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    current_user: dict = Depends(get_current_user)
):
    try:
        os.makedirs("documents", exist_ok=True)
        file_path = os.path.join("documents", file.filename)
        
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Determine file type
        file_ext = os.path.splitext(file.filename)[1].lower().replace('.', '')
        file_type = file_ext if file_ext in ['pdf', 'docx', 'csv', 'json', 'txt', 'md'] else 'txt'
        
        conn = get_db_connection()
        c = conn.cursor()
        c.execute("INSERT INTO documents (user_id, filename, file_path, file_type) VALUES (?, ?, ?, ?)", 
                  (current_user["id"], file.filename, file_path, file_type))
        document_id = c.lastrowid
        conn.commit()
        conn.close()
        
        try:
            text = extract_text_from_file(contents, file.filename, file_type)
            chunks_count = process_document(
                text, 
                {"filename": file.filename, "user_id": current_user["id"], "file_type": file_type},
                str(current_user["id"]),
                file_type
            )
            
            conn = get_db_connection()
            c = conn.cursor()
            c.execute("UPDATE documents SET processed = TRUE WHERE id = ?", (document_id,))
            conn.commit()
            conn.close()
            
            return DocumentUploadResponse(
                message=f"Document processed successfully with {chunks_count} chunks",
                document_id=document_id,
                chunks_count=chunks_count
            )
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            return DocumentUploadResponse(
                message=f"Failed to process document: {str(e)}",
                document_id=document_id,
                error=str(e)
            )
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload document: {str(e)}")

@app.post("/chat")
async def chat(
    message: ChatMessage,
    current_user: dict = Depends(get_current_user)
):
    conn = get_db_connection()
    c = conn.cursor()
    
    try:
        if not message.conversation_id:
            c.execute("INSERT INTO conversations (user_id, title) VALUES (?, ?)", 
                     (current_user["id"], message.content[:50]))
            conversation_id = c.lastrowid
        else:
            c.execute("SELECT id FROM conversations WHERE id = ? AND user_id = ?", 
                     (message.conversation_id, current_user["id"]))
            if not c.fetchone():
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation_id = message.conversation_id
        
        c.execute("INSERT INTO messages (conversation_id, content, is_user) VALUES (?, ?, ?)", 
                  (conversation_id, message.content, True))
        conn.commit()
        
        # Get conversation history for context and topic detection
        c.execute("SELECT content, is_user FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 10", 
                  (conversation_id,))
        history_messages = c.fetchall()
        history = [
            {"content": msg[0], "is_user": bool(msg[1])} 
            for msg in reversed(history_messages)
        ]
        
        # Detect topic change
        topic_change = detect_topic_change(message.content, history)
        
        # Query RAG for relevant context
        context_results = query_rag(message.content, str(current_user["id"]))
        context = context_results['documents'][0] if context_results['documents'] else []
        
        # Prepare response
        if topic_change and context:
            response_content = f"I notice we're changing topics. Based on your documents, I found some relevant information:\n\n"
        else:
            response_content = ""
        
        try:
            # Generate response using the enhanced prompt
            full_response = generate_response(message.content, context, history, topic_change)
            response_content += full_response
        except Exception as e:
            response_content += f"Sorry, I encountered an error: {str(e)}"
        
        c.execute("INSERT INTO messages (conversation_id, content, is_user) VALUES (?, ?, ?)", 
                  (conversation_id, response_content, False))
        conn.commit()
        
        return {
            "response": response_content,
            "conversation_id": conversation_id,
            "topic_change_detected": topic_change
        }
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")
    finally:
        conn.close()

@app.post("/chat/stream")
async def chat_stream(
    message: ChatMessage,
    current_user: dict = Depends(get_current_user)
):
    """Streaming chat endpoint"""
    conn = get_db_connection()
    c = conn.cursor()
    
    try:
        if not message.conversation_id:
            c.execute("INSERT INTO conversations (user_id, title) VALUES (?, ?)", 
                     (current_user["id"], message.content[:50]))
            conversation_id = c.lastrowid
        else:
            c.execute("SELECT id FROM conversations WHERE id = ? AND user_id = ?", 
                     (message.conversation_id, current_user["id"]))
            if not c.fetchone():
                raise HTTPException(status_code=404, detail="Conversation not found")
            conversation_id = message.conversation_id
        
        c.execute("INSERT INTO messages (conversation_id, content, is_user) VALUES (?, ?, ?)", 
                  (conversation_id, message.content, True))
        conn.commit()
        
        # Get conversation history
        c.execute("SELECT content, is_user FROM messages WHERE conversation_id = ? ORDER BY created_at DESC LIMIT 10", 
                  (conversation_id,))
        history_messages = c.fetchall()
        history = [
            {"content": msg[0], "is_user": bool(msg[1])} 
            for msg in reversed(history_messages)
        ]
        
        # Query RAG for relevant context
        context_results = query_rag(message.content, str(current_user["id"]))
        context = context_results['documents'][0] if context_results['documents'] else []
        
        # Create a streaming response
        return StreamingResponse(
            generate_streaming_response(message.content, context, history),
            media_type="text/plain"
        )
    except Exception as e:
        conn.rollback()
        raise HTTPException(status_code=500, detail=f"Streaming chat error: {str(e)}")
    finally:
        conn.close()

@app.get("/conversations")
async def get_conversations(current_user: dict = Depends(get_current_user)):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT id, title, created_at FROM conversations WHERE user_id = ? ORDER BY created_at DESC", 
                  (current_user["id"],))
        conversations = c.fetchall()
        return [{"id": conv[0], "title": conv[1], "created_at": conv[2]} for conv in conversations]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversations: {str(e)}")
    finally:
        conn.close()

@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    current_user: dict = Depends(get_current_user)
):
    conn = get_db_connection()
    c = conn.cursor()
    try:
        c.execute("SELECT id, title FROM conversations WHERE id = ? AND user_id = ?", 
                  (conversation_id, current_user["id"]))
        conversation = c.fetchone()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        c.execute("SELECT id, content, is_user, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC", 
                  (conversation_id,))
        messages = c.fetchall()
        
        return {
            "conversation": {"id": conversation[0], "title": conversation[1]},
            "messages": [{"id": msg[0], "content": msg[1], "is_user": bool(msg[2]), "created_at": msg[3]} for msg in messages]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching conversation: {str(e)}")
    finally:
        conn.close()

@app.post("/send-email")
async def send_email_endpoint(
    email_request: EmailRequest,
    current_user: dict = Depends(get_current_user)
):
    """Send email using templates to the specified recipient"""
    success = send_email(
        email_request.template_name,
        email_request.recipient_email,
        email_request.parameters
    )
    
    if success:
        return {"message": f"Email sent successfully to {email_request.recipient_email}"}
    else:
        raise HTTPException(status_code=500, detail="Failed to send email")

@app.post("/conversations/{conversation_id}/send-summary")
async def send_conversation_summary(
    conversation_id: int,
    recipient_email: EmailStr,
    current_user: dict = Depends(get_current_user)
):
    """Send a summary of a conversation to the specified email"""
    # Get conversation details
    conn = get_db_connection()
    c = conn.cursor()
    
    try:
        # Verify conversation belongs to user
        c.execute("SELECT id, title FROM conversations WHERE id = ? AND user_id = ?", 
                  (conversation_id, current_user["id"]))
        conversation = c.fetchone()
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Get conversation messages
        c.execute("SELECT content, is_user, created_at FROM messages WHERE conversation_id = ? ORDER BY created_at ASC", 
                  (conversation_id,))
        messages = c.fetchall()
        
        # Create summary
        conversation_summary = f"Conversation: {conversation[1]}\n\n"
        for msg in messages:
            role = "User" if msg[1] else "Assistant"
            conversation_summary += f"{role} ({msg[2]}): {msg[0]}\n\n"
        
        # Send email with summary
        success = send_email(
            "summary",
            recipient_email,
            {
                "name": current_user["full_name"],
                "summary": conversation_summary
            }
        )
        
        if success:
            return {"message": f"Conversation summary sent to {recipient_email}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to send email")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending summary: {str(e)}")
    finally:
        conn.close()

@app.get("/email-configuration")
async def check_email_configuration(current_user: dict = Depends(get_current_user)):
    """Check if email is properly configured"""
    if EMAIL_USER == "your-email@gmail.com" or EMAIL_PASSWORD == "your-app-password":
        return {
            "configured": False,
            "message": "Email not configured. Please update EMAIL_USER and EMAIL_PASSWORD in the code."
        }
    
    # Test connection
    try:
        server = smtplib.SMTP(EMAIL_HOST, EMAIL_PORT)
        server.ehlo()
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.quit()
        return {"configured": True, "message": "Email configuration is valid"}
    except Exception as e:
        return {"configured": False, "message": f"Email configuration error: {str(e)}"}

@app.get("/users/me")
async def read_users_me(current_user: dict = Depends(get_current_user)):
    return current_user

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "chromadb": chroma_client is not None,
        "embedding_model": embedding_model is not None,
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)