# app.py - Secure Home Insurance Chatbot with Admin Dashboard
# Streamlit + Groq + Lead Management
# Run: streamlit run app.py

import os
import uuid
import json
import streamlit as st
from dotenv import load_dotenv
import re
import logging
from datetime import datetime
from typing import Optional, Dict, List
import pandas as pd
import sqlite3
import hashlib
from pathlib import Path
import io

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# ──────────────────────────────────────────────────────────────
# Setup & Configuration
# ──────────────────────────────────────────────────────────────

load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
DB_FILE = "insurance_leads.db"
ADMIN_USERNAME = "admin"

# Set admin password in .env: ADMIN_PASSWORD_HASH=your_sha256_hash_here
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in environment variables")
    st.stop()

# Session state initialization
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "user_leads" not in st.session_state:
    st.session_state.user_leads = []
if "show_quick_form" not in st.session_state:
    st.session_state.show_quick_form = False
if "show_admin" not in st.session_state:
    st.session_state.show_admin = False

# ──────────────────────────────────────────────────────────────
# Database Setup
# ──────────────────────────────────────────────────────────────

def init_database():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS leads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            session_id TEXT NOT NULL,
            name TEXT,
            email TEXT,
            phone TEXT,
            location TEXT,
            home_value_range TEXT,
            interest_level TEXT,
            conversation_summary TEXT,
            ip_address TEXT,
            user_agent TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS admin_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            admin_user TEXT NOT NULL,
            details TEXT
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("Database initialized")

def save_lead_to_db(lead_data: Dict):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        ip = headers.get("X-Forwarded-For", "unknown")
        ua = headers.get("User-Agent", "unknown")
    except:
        ip = ua = "unknown"

    c.execute('''
        INSERT INTO leads 
        (timestamp, session_id, name, email, phone, location, 
         home_value_range, interest_level, conversation_summary, ip_address, user_agent)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        lead_data.get('timestamp', datetime.now().isoformat()),
        lead_data.get('session_id', st.session_state.session_id),
        lead_data.get('name'),
        lead_data.get('email'),
        lead_data.get('phone'),
        lead_data.get('location'),
        lead_data.get('home_value_range'),
        lead_data.get('interest_level', 'low'),
        lead_data.get('conversation_summary', ''),
        ip,
        ua
    ))
    conn.commit()
    conn.close()
    logger.info(f"Lead saved: {lead_data.get('email', 'anonymous')}")

def get_all_leads() -> pd.DataFrame:
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM leads ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_lead_count() -> int:
    if not os.path.exists(DB_FILE):
        return 0
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM leads")
    count = c.fetchone()[0]
    conn.close()
    return count

def log_admin_action(action: str, details: str = ""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        INSERT INTO admin_logs (timestamp, action, admin_user, details)
        VALUES (?, ?, ?, ?)
    ''', (datetime.now().isoformat(), action, ADMIN_USERNAME, details))
    conn.commit()
    conn.close()

init_database()

# ──────────────────────────────────────────────────────────────
# LLM Setup
# ──────────────────────────────────────────────────────────────

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.3,
    groq_api_key=GROQ_API_KEY,
    max_tokens=1024,
)

system_prompt = """
You are a professional, helpful home insurance assistant.
Your goals:
- Provide clear, accurate information about home insurance
- Explain coverage types, policies, exclusions, and claims
- Guide users toward getting personalized quotes when appropriate
- Collect contact info naturally only if the user shows strong interest
- Always remain polite, calm, and trustworthy
- Remind users that you are not a licensed agent and official quotes require professional consultation
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

runnable = prompt | llm

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return StreamlitChatMessageHistory(key=f"history_{session_id}")

chain = RunnableWithMessageHistory(
    runnable,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history"
)

# ──────────────────────────────────────────────────────────────
# Helper Functions (unchanged core logic – only minor cleanup)
# ──────────────────────────────────────────────────────────────

def check_admin_password(password: str) -> bool:
    if not ADMIN_PASSWORD_HASH:
        return password == "admin123"  # fallback for testing
    return hashlib.sha256(password.encode()).hexdigest() == ADMIN_PASSWORD_HASH

def extract_contact_info(text: str) -> Dict:
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    return {'email': emails[0] if emails else None, 'phone': phones[0] if phones else None}

def analyze_conversation_for_lead(messages: List[Dict]) -> Dict:
    full_convo = " ".join([m['content'] for m in messages[-8:]])
    lead = {
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.session_id,
        'name': None,
        'email': None,
        'phone': None,
        'location': None,
        'home_value_range': None,
        'interest_level': 'low',
        'conversation_summary': full_convo[:600]
    }
    contact = extract_contact_info(full_convo)
    lead.update(contact)

    # Basic name guess
    for msg in messages:
        if "name is" in msg['content'].lower():
            parts = msg['content'].lower().split("name is")
            if len(parts) > 1:
                name_part = parts[1].split()[0].strip(".,!?")
                if len(name_part) > 1:
                    lead['name'] = name_part.title()
                    break

    # Interest scoring
    high_triggers = ["email me", "call me", "contact me", "quote now", "send quote"]
    med_triggers = ["quote", "price", "cost", "premium", "coverage", "policy"]
    score = sum(2 for t in high_triggers if t in full_convo.lower()) + \
            sum(1 for t in med_triggers if t in full_convo.lower())
    if score >= 4:
        lead['interest_level'] = 'high'
    elif score >= 2:
        lead['interest_level'] = 'medium'

    return lead

# ──────────────────────────────────────────────────────────────
# Admin Dashboard (kept mostly as-is, minor layout polish)
# ──────────────────────────────────────────────────────────────

def show_admin_dashboard():
    st.title("🔐 Admin Dashboard")
    st.markdown(f"**Logged in as:** {ADMIN_USERNAME} • Last active: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Leads", "📈 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.header("Lead Management")
        df = get_all_leads()
        if not df.empty:
            st.metric("Total Leads Captured", len(df))
            col1, col2 = st.columns(2)
            with col1:
                interest_filter = st.selectbox("Filter by Interest Level", ["All", "High", "Medium", "Low"])
            with col2:
                date_filter = st.date_input("Filter by Date", value=None)
            
            if interest_filter != "All":
                df = df[df['interest_level'] == interest_filter.lower()]
            if date_filter:
                df = df[df['timestamp'].str.contains(str(date_filter))]
            
            st.dataframe(
                df.drop(['ip_address', 'user_agent'], axis=1, errors='ignore'),
                use_container_width=True,
                hide_index=True
            )
            
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Export Leads (CSV)",
                csv,
                f"leads_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
            
            with st.expander("🗑️ Danger Zone – Delete Data", expanded=False):
                st.warning("This cannot be undone.")
                if st.button("Delete ALL Leads", type="primary"):
                    conn = sqlite3.connect(DB_FILE)
                    conn.execute("DELETE FROM leads")
                    conn.commit()
                    conn.close()
                    log_admin_action("delete_all_leads", "All leads removed")
                    st.success("All leads deleted")
                    st.rerun()
        else:
            st.info("No leads recorded yet.")

    with tab2:
        st.header("Analytics Overview")
        df = get_all_leads()
        if not df.empty:
            col1, col2, col3 = st.columns(3)
            col1.metric("High Interest", len(df[df['interest_level'] == 'high']))
            col2.metric("With Email", df['email'].notna().sum())
            col3.metric("Today", df[df['timestamp'].str.contains(datetime.now().strftime('%Y-%m-%d'))].shape[0])

            st.subheader("Interest Level Breakdown")
            st.bar_chart(df['interest_level'].value_counts())

            st.subheader("Leads Over Time")
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            st.line_chart(df.groupby('date').size())
        else:
            st.info("No analytics data yet.")

    with tab3:
        st.header("Settings & Logs")
        st.info(f"Database: {DB_FILE} • Size: {os.path.getsize(DB_FILE)/1024:.1f} KB" if os.path.exists(DB_FILE) else "No database found")
        
        if st.button("Backup Database"):
            if os.path.exists(DB_FILE):
                with open(DB_FILE, "rb") as f:
                    st.download_button("Download Backup (.db)", f, f"backup_{datetime.now():%Y%m%d}.db", "application/x-sqlite3")
            log_admin_action("database_backup_attempt")

        st.subheader("Recent Admin Actions")
        conn = sqlite3.connect(DB_FILE)
        logs = pd.read_sql_query("SELECT * FROM admin_logs ORDER BY timestamp DESC LIMIT 30", conn)
        conn.close()
        if not logs.empty:
            st.dataframe(logs, use_container_width=True, hide_index=True)
        else:
            st.info("No admin actions logged yet.")

    if st.sidebar.button("🚪 Logout", type="primary"):
        st.session_state.admin_logged_in = False
        log_admin_action("logout")
        st.rerun()

# ──────────────────────────────────────────────────────────────
# Page Config & Modern Styling
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Home Insurance Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Premium header */
    .main-header {
        text-align: center;
        padding: 2.5rem 1rem;
        background: linear-gradient(135deg, #f0f7ff 0%, #e0f2fe 100%);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);
    }
    .main-header h1 {
        color: #1e40af;
        font-size: 2.8rem;
        margin: 0 0 0.5rem 0;
    }
    .main-header p {
        color: #1e3a8a;
        font-size: 1.25rem;
        margin: 0;
    }

    /* Chat styling */
    .stChatMessage {
        border-radius: 18px !important;
        padding: 1.1rem 1.4rem !important;
        margin-bottom: 1rem !important;
    }
    .stChatMessage.user {
        background-color: #dbeafe !important;
        border-radius: 18px 18px 4px 18px !important;
    }
    .stChatMessage.assistant {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 18px 18px 18px 4px !important;
    }

    /* Sidebar polish */
    section[data-testid="stSidebar"] {
        background-color: #f9fafb !important;
        border-right: 1px solid #e5e7eb;
    }
    .sidebar .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        font-weight: 600;
        margin-bottom: 0.8rem;
        width: 100%;
    }
    .sidebar .stButton > button:hover {
        background: #2563eb;
    }

    /* Footer */
    .footer-col {
        text-align: center;
        font-size: 0.95rem;
        color: #4b5563;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0;">
        <h1 style="font-size: 2.5rem; margin: 0;">🏠</h1>
        <h2 style="margin: 0.5rem 0 1rem 0;">Insurance Assistant</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    with st.expander("🔒 Admin Access", expanded=False):
        if not st.session_state.admin_logged_in:
            admin_pw = st.text_input("Admin Password", type="password")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", use_container_width=True):
                    if check_admin_password(admin_pw):
                        st.session_state.admin_logged_in = True
                        log_admin_action("login_success")
                        st.success("Login successful")
                        st.rerun()
                    else:
                        st.error("Incorrect password")
            with col2:
                if st.button("Clear", use_container_width=True):
                    st.rerun()
        else:
            st.success("✅ Admin logged in")
            if st.button("Open Dashboard", use_container_width=True):
                st.session_state.show_admin = True
                st.rerun()
    
    st.divider()
    
    st.header("Quick Actions")
    if st.button("🔄 Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    if st.button("📝 Request Quick Quote", use_container_width=True):
        st.session_state.show_quick_form = True
        st.rerun()
    
    st.divider()
    
    st.caption("""
    **Important:** This assistant provides general information only.  
    For official quotes or advice, consult a licensed insurance professional.
    """)
    st.caption("Your privacy is protected • Data handled securely")

# ──────────────────────────────────────────────────────────────
# Main Area
# ──────────────────────────────────────────────────────────────

if st.session_state.get("show_admin", False) and st.session_state.admin_logged_in:
    show_admin_dashboard()
    st.stop()

# Premium header + Impact verification (placed early for crawlers)
st.markdown("""
<div class="main-header">
    <h1>Home Insurance Assistant</h1>
    <p>Clear answers • Coverage comparisons • Personalized guidance</p>
</div>

<!-- Impact verification - must be in body text, visible to crawlers -->
<div style="font-size: 12px; color: #ccc; text-align: center; margin: 10px 0;">
Impact-Site-Verification: f2aacffd-dcf7-4f0f-84f8-0df56150dc65
</div>
""", unsafe_allow_html=True)

# Quick Quote Form
if st.session_state.show_quick_form:
    st.subheader("📋 Quick Quote Request")
    with st.form("quick_quote"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name")
            email = st.text_input("Email Address")
        with col2:
            location = st.text_input("City, State")
            home_value = st.selectbox(
                "Approximate Home Value",
                ["Under $200,000", "$200,000–$500,000", "$500,000–$1,000,000", "Over $1,000,000", "Not sure"]
            )
        
        if st.form_submit_button("Submit Request", type="primary"):
            if email and "@" in email:
                lead = {
                    'name': name,
                    'email': email,
                    'location': location,
                    'home_value_range': home_value,
                    'interest_level': 'high',
                    'conversation_summary': 'Quick quote form submission'
                }
                save_lead_to_db(lead)
                st.session_state.user_leads.append(lead)
                st.success("Thank you! A representative will reach out soon with personalized options.")
                st.session_state.show_quick_form = False
                st.rerun()
            else:
                st.error("Please provide a valid email address")

# Chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about coverage, quotes, claims..."):
    prompt = prompt.strip()
    if len(prompt) < 3:
        st.warning("Please type a more detailed question")
    elif len(prompt) > 1000:
        st.warning("Message too long – please shorten it")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Preparing response..."):
                try:
                    response = chain.invoke(
                        {"input": prompt},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    reply = response.content
                    
                    contact = extract_contact_info(prompt + " " + reply)
                    if contact['email'] or contact['phone']:
                        st.success("Contact information received – we'll follow up shortly.")
                        lead = analyze_conversation_for_lead(st.session_state.messages)
                        lead.update(contact)
                        save_lead_to_db(lead)
                    
                    if any(w in prompt.lower() for w in ["quote", "cost", "price", "premium", "rate"]):
                        if not (contact['email'] or contact['phone']):
                            reply += "\n\n💡 For a personalized quote, feel free to share your email or phone."
                    
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                
                except Exception as e:
                    logger.error(f"Response error: {e}")
                    st.error("Sorry, something went wrong. Please try again or use the quick quote form.")
                    st.session_state.messages.append({"role": "assistant", "content": "I apologize for the issue. Try rephrasing or use the form above."})

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown('<div class="footer-col">🔒 Secure & Private<br>Your information is protected</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="footer-col">📞 Live Support<br>9am–6pm CST, Mon–Fri</div>', unsafe_allow_html=True)
with col3:
    if st.session_state.admin_logged_in:
        st.markdown('<div class="footer-col">👑 Admin Mode Active</div>', unsafe_allow_html=True)
        if st.button("View Dashboard", type="secondary"):
            st.session_state.show_admin = True
            st.rerun()
    else:
        st.markdown('<div class="footer-col">🏠 Professional Home Insurance Guidance</div>', unsafe_allow_html=True)

# Hidden admin indicator
if st.session_state.admin_logged_in:
    st.sidebar.markdown('<div style="background:#dc2626;color:white;padding:0.4rem 0.8rem;border-radius:6px;font-size:0.8rem;text-align:center;">ADMIN ACTIVE</div>', unsafe_allow_html=True)
