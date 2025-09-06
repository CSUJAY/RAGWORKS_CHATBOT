# 🚀 RAGWorks Project - Complete Setup Guide

Here's the comprehensive `README.md` file for your RAGWorks project:

```markdown
# RAGWorks Project

A complete RAG (Retrieval-Augmented Generation) powered chat application with user authentication, document processing, and email integration.

## 📋 Prerequisites

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection (for model downloads and email functionality)

### Software Dependencies
- Ollama (for running the LLM locally)
- Python packages (listed in requirements.txt)

## 🛠 Installation & Setup

### 1. Install Python 3.8+
```bash
# Check if Python is installed
python --version
python -m pip install --upgrade pip
```

### 2. Install Required Dependencies
Create a `requirements.txt` file with the following content:
```txt
fastapi==0.104.1
uvicorn[standard]==0.24.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
chromadb==0.4.15
sentence-transformers==2.2.2
ollama==0.1.4
streamlit==1.28.1
requests==2.31.0
pypdf2==3.0.1
python-docx==1.1.0
python-dotenv==1.0.0
email-validator==2.0.0
```

Then install them:
```bash
pip install -r requirements.txt
```

### 3. Install and Setup Ollama
```bash
# Download and install Ollama
# On macOS/Linux:
curl -fsSL https://ollama.ai/install.sh | sh

# On Windows, download from https://ollama.ai/

# Pull the required model
ollama pull llama2
```

### 4. Configure Email (Optional but Recommended)
Create a `.env` file in your backend directory:
```env
SECRET_KEY=your-super-secret-key-here
EMAIL_USER=your-email@gmail.com
EMAIL_PASSWORD=your-app-password
```

**To get Gmail App Password:**
1. Enable 2-factor authentication on your Gmail account
2. Go to Google Account → Security → App passwords
3. Generate a password for "Mail" and use that in the `.env` file

### 5. Directory Structure Setup
```bash
mkdir -p backend documents chroma_db
```

## 🎯 Inputs Needed During Execution

### 1. First Run Setup
The application will automatically:
- Create SQLite database (`chat_app.db`)
- Initialize ChromaDB vector store
- Create default email templates
- Set up database tables

### 2. User Registration/Login
**Inputs needed:**
- Email address
- Password
- Full name (for registration)

### 3. Document Upload
**Supported file types:**
- PDF (.pdf)
- Word documents (.docx)
- Text files (.txt, .md)
- CSV files (.csv)
- JSON files (.json)

**Input during upload:**
- File selection through UI
- Automatic file type detection

### 4. Chat Interface
**Inputs needed:**
- Message text input
- Conversation selection (optional)
- Streaming toggle preference

### 5. Email Features
**Inputs for sending emails:**
- Recipient email address
- Template selection (welcome/summary)
- Custom parameters (name, summary content)

## 🚀 Running the Application

### 1. Start the Backend (FastAPI)
```bash
cd backend
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```
Backend will be available at: http://localhost:8000  
API documentation: http://localhost:8000/docs

### 2. Start the Frontend (Streamlit)
```bash
cd frontend
streamlit run streamlit_app.py
```
Frontend will be available at: http://localhost:8501

## 🔧 Configuration

### Critical: Update Email Credentials
In `app.py`, change these lines:
```python
EMAIL_USER = "your-actual-email@gmail.com"  # Change this
EMAIL_PASSWORD = "your-actual-app-password"  # Change this
```

### Optional: Modify Default Settings
You can change:
- `SECRET_KEY` for JWT tokens
- `ACCESS_TOKEN_EXPIRE_MINUTES` for session duration
- `EMAIL_HOST` and `EMAIL_PORT` for different email providers
- Embedding model in `SentenceTransformer('all-MiniLM-L6-v2')`

## 🐛 Troubleshooting

### 1. Ollama Not Found
```bash
# Check if Ollama is running
ollama list

# If not installed, download from https://ollama.ai/
```

### 2. ChromaDB Issues
```bash
# Delete and recreate chroma_db directory if needed
rm -rf chroma_db/
```

### 3. Email Not Working
- Verify app password is correct
- Check if less secure apps are enabled (for Gmail)
- Try different email provider if needed

### 4. Port Already in Use
```bash
# Kill processes on ports 8000 or 8501
# On macOS/Linux:
lsof -ti:8000 | xargs kill
lsof -ti:8501 | xargs kill

# On Windows:
netstat -ano | findstr :8000
taskkill /PID <pid> /F
```

## 📊 Health Check

After starting, visit: http://localhost:8000/health  
Should return:
```json
{
  "status": "healthy",
  "chromadb": true,
  "embedding_model": true,
  "timestamp": "2023-12-07T10:30:00.123456"
}
```

## 🎉 Success Indicators

When everything is working correctly, you should see:
1. ✅ Backend server running on port 8000
2. ✅ Streamlit frontend on port 8501
3. ✅ Ability to register/login users
4. ✅ Document upload functionality
5. ✅ Chat interface with RAG responses
6. ✅ Email sending capability (if configured)

## 📁 Project Structure
```
├── backend/
│   ├── app.py              # FastAPI application
│   ├── auth.py             # Authentication utilities
│   ├── database.py         # Database connection & models
│   ├── chroma_utils.py     # ChromaDB utilities
│   ├── email_utils.py      # Email sending utilities
│   └── .env               # Environment variables
├── frontend/
│   └── streamlit_app.py    # Streamlit UI application
├── documents/              # Uploaded documents storage
├── chroma_db/             # ChromaDB vector store
└── requirements.txt       # Python dependencies
```





This setup provides a complete RAG-powered chat application with user authentication, document processing, and email integration!
```

This README.md file includes all the essential information for setting up and running your RAGWorks project, organized in a clear and user-friendly manner. It covers prerequisites, installation steps, configuration, troubleshooting, and success indicators to help users get started quickly.
