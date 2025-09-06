import streamlit as st
import requests
import json
from datetime import datetime
import time

# Configuration
BACKEND_URL = "http://127.0.0.1:8000"

# Initialize session state
if "token" not in st.session_state:
    st.session_state.token = None
if "user" not in st.session_state:
    st.session_state.user = None
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "current_conversation" not in st.session_state:
    st.session_state.current_conversation = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "streaming" not in st.session_state:
    st.session_state.streaming = True  # Default to streaming enabled
if "streaming_response" not in st.session_state:
    st.session_state.streaming_response = ""
if "streaming_active" not in st.session_state:
    st.session_state.streaming_active = False
if "email_conversation_id" not in st.session_state:
    st.session_state.email_conversation_id = None
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False
if "conversations_loaded" not in st.session_state:
    st.session_state.conversations_loaded = False

def login(email, password):
    try:
        response = requests.post(
            f"{BACKEND_URL}/token",
            data={"username": email, "password": password},
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        if response.status_code == 200:
            st.session_state.token = response.json()["access_token"]
            # Get user info
            user_response = requests.get(
                f"{BACKEND_URL}/users/me",
                headers={"Authorization": f"Bearer {st.session_state.token}"}
            )
            if user_response.status_code == 200:
                st.session_state.user = user_response.json()
            st.success("Login successful!")
            # Load conversations after successful login
            st.session_state.conversations = get_conversations()
            st.session_state.conversations_loaded = True
            return True
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Login failed: {error_detail}")
            return False
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def register(email, password, full_name):
    try:
        response = requests.post(
            f"{BACKEND_URL}/register",
            json={"email": email, "password": password, "full_name": full_name}
        )
        if response.status_code == 200:
            st.success("Registration successful! Please login.")
            return True
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Registration failed: {error_detail}")
            return False
    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False

def upload_document(file):
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        files = {"file": (file.name, file.getvalue(), file.type)}
        response = requests.post(
            f"{BACKEND_URL}/upload-document",
            files=files,
            headers=headers
        )
        if response.status_code == 200:
            result = response.json()
            st.success(f"Document uploaded successfully! {result.get('message', '')}")
            st.session_state.document_uploaded = True
            return True
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Upload failed: {error_detail}")
            return False
    except Exception as e:
        st.error(f"Upload error: {str(e)}")
        return False

def send_message(message, conversation_id=None, stream=False):
    try:
        headers = {
            "Authorization": f"Bearer {st.session_state.token}",
            "Content-Type": "application/json"
        }
        data = {"content": message, "conversation_id": conversation_id}
        
        if stream:
            # Use streaming endpoint
            response = requests.post(
                f"{BACKEND_URL}/chat/stream",
                json=data,
                headers=headers,
                stream=True
            )
            return response
        else:
            # Use regular endpoint
            response = requests.post(
                f"{BACKEND_URL}/chat",
                json=data,
                headers=headers
            )
            if response.status_code == 200:
                return response.json()
            else:
                error_detail = response.json().get('detail', 'Unknown error')
                st.error(f"Chat error: {error_detail}")
                return None
    except Exception as e:
        st.error(f"Chat error: {str(e)}")
        return None

def get_conversations():
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(
            f"{BACKEND_URL}/conversations",
            headers=headers
        )
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Failed to get conversations: {error_detail}")
            return []
    except Exception as e:
        st.error(f"Conversations error: {str(e)}")
        return []

def get_conversation(conversation_id):
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        response = requests.get(
            f"{BACKEND_URL}/conversations/{conversation_id}",
            headers=headers
        )
        if response.status_code == 200:
            return response.json()
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Failed to get conversation: {error_detail}")
            return None
    except Exception as e:
        st.error(f"Conversation error: {str(e)}")
        return None

def send_conversation_summary(conversation_id, recipient_email):
    try:
        headers = {
            "Authorization": f"Bearer {st.session_state.token}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{BACKEND_URL}/conversations/{conversation_id}/send-summary",
            json={"recipient_email": recipient_email},
            headers=headers
        )
        if response.status_code == 200:
            st.success("Conversation summary sent successfully!")
            return True
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Email failed: {error_detail}")
            return False
    except Exception as e:
        st.error(f"Email error: {str(e)}")
        return False

def send_custom_email(template_name, recipient_email, parameters):
    try:
        headers = {
            "Authorization": f"Bearer {st.session_state.token}",
            "Content-Type": "application/json"
        }
        data = {
            "template_name": template_name,
            "recipient_email": recipient_email,
            "parameters": parameters
        }
        response = requests.post(
            f"{BACKEND_URL}/send-email",
            json=data,
            headers=headers
        )
        if response.status_code == 200:
            st.success("Email sent successfully!")
            return True
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            st.error(f"Email failed: {error_detail}")
            return False
    except Exception as e:
        st.error(f"Email error: {str(e)}")
        return False

def process_streaming_response(response, conversation_id=None):
    """Process streaming response and update UI in real-time"""
    st.session_state.streaming_active = True
    st.session_state.streaming_response = ""
    
    message_placeholder = st.empty()
    full_response = ""
    
    try:
        for chunk in response.iter_content(chunk_size=1024, decode_unicode=True):
            if chunk:
                full_response += chunk
                st.session_state.streaming_response = full_response
                message_placeholder.markdown(full_response + "▌")
        
        # Final update without the cursor
        message_placeholder.markdown(full_response)
        
        # Add to messages
        st.session_state.messages.append({"content": full_response, "is_user": False})
        
        # If this was a new conversation, get the conversation ID
        if conversation_id is None:
            # Refresh conversations to get the new one
            conversations = get_conversations()
            if conversations:
                st.session_state.current_conversation = conversations[0]["id"]
                st.session_state.conversations = conversations
                
    except Exception as e:
        st.error(f"Streaming error: {str(e)}")
    finally:
        st.session_state.streaming_active = False
        st.session_state.streaming_response = ""

# UI Components
def login_form():
    with st.form("login_form"):
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")
        
        if submitted:
            if login(email, password):
                st.rerun()

def register_form():
    with st.form("register_form"):
        st.subheader("Register")
        email = st.text_input("Email")
        full_name = st.text_input("Full Name")
        password = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        submitted = st.form_submit_button("Register")
        
        if submitted:
            if password == confirm_password:
                if register(email, password, full_name):
                    # Clear form after successful registration
                    st.rerun()
            else:
                st.error("Passwords do not match")

def email_interface():
    st.header("📧 Send Email")
    
    with st.form("email_form"):
        template = st.selectbox("Template", ["welcome", "summary"])
        recipient = st.text_input("Recipient Email")
        
        if template == "welcome":
            name = st.text_input("Recipient Name", "John Doe")
            params = {"name": name}
        else:
            summary = st.text_area("Conversation Summary", "Summary of our conversation...")
            name = st.text_input("Recipient Name", "John Doe")
            params = {"name": name, "summary": summary}
        
        if st.form_submit_button("Send Email"):
            send_custom_email(template, recipient, params)

def conversation_email_interface(conversation_id):
    st.header(f"📧 Email Conversation #{conversation_id}")
    
    with st.form("conversation_email_form"):
        recipient = st.text_input("Recipient Email")
        
        if st.form_submit_button("Send Conversation Summary"):
            if send_conversation_summary(conversation_id, recipient):
                st.session_state.email_conversation_id = None
                st.rerun()
    
    if st.button("Back to Chat"):
        st.session_state.email_conversation_id = None
        st.rerun()

def chat_interface():
    st.title("🤖 Advanced LLM Chat with RAG")
    
    # Sidebar
    with st.sidebar:
        st.header("💬 Conversations")
        
        if st.button("➕ New Conversation", use_container_width=True):
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.session_state.streaming_response = ""
            st.rerun()
        
        # Refresh conversations button
        if st.button("🔄 Refresh Conversations", use_container_width=True):
            st.session_state.conversations = get_conversations()
            st.rerun()
        
        if not st.session_state.conversations_loaded:
            st.session_state.conversations = get_conversations()
            st.session_state.conversations_loaded = True
        
        if st.session_state.conversations:
            st.write("**Your Conversations:**")
            for conv in st.session_state.conversations:
                col1, col2 = st.columns([3, 1])
                with col1:
                    if st.button(f"{conv['title']}", key=f"conv_{conv['id']}", use_container_width=True):
                        st.session_state.current_conversation = conv["id"]
                        conversation_data = get_conversation(conv["id"])
                        if conversation_data:
                            st.session_state.messages = conversation_data["messages"]
                        st.rerun()
                with col2:
                    if st.button("📧", key=f"email_{conv['id']}", help="Email this conversation"):
                        st.session_state.email_conversation_id = conv["id"]
                        st.rerun()
        else:
            st.info("No conversations yet. Start chatting to create one!")
        
        st.divider()
        
        st.header("📁 Document Management")
        uploaded_file = st.file_uploader("Upload a document", type=["txt", "pdf", "docx", "md", "csv", "json"])
        if uploaded_file is not None:
            if st.button("Process Document", use_container_width=True):
                with st.spinner("Processing document..."):
                    if upload_document(uploaded_file):
                        st.rerun()
        
        st.divider()
        
        st.header("⚙️ Settings")
        
        # Streaming option
        st.session_state.streaming = st.checkbox("Enable Streaming Responses", value=True)
        
        # Email tools
        if st.checkbox("Show Email Tools"):
            email_interface()
        
        st.divider()
        
        # User info
        if st.session_state.user:
            st.write(f"**Logged in as:** {st.session_state.user['email']}")
            st.write(f"**Name:** {st.session_state.user.get('full_name', 'N/A')}")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.token = None
            st.session_state.user = None
            st.session_state.conversations = []
            st.session_state.current_conversation = None
            st.session_state.messages = []
            st.session_state.streaming_response = ""
            st.session_state.conversations_loaded = False
            st.rerun()
    
    # Main chat area
    if st.session_state.email_conversation_id:
        conversation_email_interface(st.session_state.email_conversation_id)
        return
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        if not st.session_state.current_conversation:
            st.header("💬 New Conversation")
        else:
            conv_title = next((conv['title'] for conv in st.session_state.conversations 
                              if conv['id'] == st.session_state.current_conversation), 'Conversation')
            st.header(f"💬 {conv_title}")
    
    with col2:
        if st.session_state.current_conversation:
            if st.button("📋 Email Summary", help="Email summary of this conversation"):
                st.session_state.email_conversation_id = st.session_state.current_conversation
                st.rerun()
    
    # Display messages
    for msg in st.session_state.messages:
        with st.chat_message("user" if msg["is_user"] else "assistant"):
            st.write(msg["content"])
    
    # Display streaming response if active
    if st.session_state.streaming_active:
        with st.chat_message("assistant"):
            st.markdown(st.session_state.streaming_response + "▌")
    
    # Chat input
    if prompt := st.chat_input("Type your message..."):
        # Add user message to UI
        with st.chat_message("user"):
            st.write(prompt)
        
        # Add to session state
        st.session_state.messages.append({"content": prompt, "is_user": True})
        
        # Send message and handle response
        if st.session_state.streaming:
            response = send_message(prompt, st.session_state.current_conversation, True)
            if response:
                process_streaming_response(response, st.session_state.current_conversation)
                st.rerun()
        else:
            response = send_message(prompt, st.session_state.current_conversation, False)
            if response:
                with st.chat_message("assistant"):
                    st.write(response["response"])
                st.session_state.messages.append({"content": response["response"], "is_user": False})
                # If this was a new conversation, update the current conversation ID
                if not st.session_state.current_conversation:
                    st.session_state.current_conversation = response["conversation_id"]
                # Refresh conversations list
                st.session_state.conversations = get_conversations()
                st.rerun()

# Main app logic
def main():
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .stButton button {
        width: 100%;
    }
    .stChatMessage {
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    div[data-testid="stVerticalBlock"] > div:has(>.stChatMessage) {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    if st.session_state.token is None:
        st.title("🤖 Advanced LLM Chat with RAG")
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            login_form()
        
        with tab2:
            register_form()
            
        st.markdown("---")
        st.info("Don't have an account? Register first, then login to access the chat interface.")
    else:
        chat_interface()

if __name__ == "__main__":
    main()