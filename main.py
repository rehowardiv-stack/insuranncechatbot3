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

# Set admin password in .env: ADMIN_PASSWORD=your_secure_password_here
# Or generate hash: python -c "import hashlib; print(hashlib.sha256('yourpassword'.encode()).hexdigest())"
ADMIN_PASSWORD_HASH = os.getenv("ADMIN_PASSWORD_HASH", "")

# Check API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()

# Initialize session states
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "admin_logged_in" not in st.session_state:
    st.session_state.admin_logged_in = False
if "user_leads" not in st.session_state:
    st.session_state.user_leads = []

# ──────────────────────────────────────────────────────────────
# Database Setup (Secure)
# ──────────────────────────────────────────────────────────────

def init_database():
    """Initialize SQLite database for secure lead storage"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Create leads table
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
    
    # Create admin logs table
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
    """Save lead to database securely"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    # Get user info if available
    try:
        from streamlit.web.server.websocket_headers import _get_websocket_headers
        headers = _get_websocket_headers()
        ip_address = headers.get("X-Forwarded-For", "unknown") if headers else "unknown"
        user_agent = headers.get("User-Agent", "unknown") if headers else "unknown"
    except:
        ip_address = "unknown"
        user_agent = "unknown"
    
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
        ip_address,
        user_agent
    ))
    
    conn.commit()
    conn.close()
    logger.info(f"Lead saved to database: {lead_data.get('email', 'No email')}")

def get_all_leads() -> pd.DataFrame:
    """Get all leads (admin only)"""
    if not os.path.exists(DB_FILE):
        return pd.DataFrame()
    
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM leads ORDER BY timestamp DESC", conn)
    conn.close()
    return df

def get_lead_count() -> int:
    """Get total lead count (for admin only)"""
    if not os.path.exists(DB_FILE):
        return 0
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM leads")
    count = c.fetchone()[0]
    conn.close()
    return count

def log_admin_action(action: str, details: str = ""):
    """Log admin activities"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    c.execute('''
        INSERT INTO admin_logs (timestamp, action, admin_user, details)
        VALUES (?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        action,
        ADMIN_USERNAME,
        details
    ))
    
    conn.commit()
    conn.close()

# Initialize database
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
You are a professional home insurance assistant designed to help users understand insurance options.
Your role:
1. Provide helpful information about home insurance
2. Answer questions about coverage, policies, and claims
3. Guide users on getting quotes
4. Collect information naturally if users are interested
5. Be warm, professional, and accurate
6. Always remind users to speak with licensed agents for official quotes

When users ask about quotes, you can ask for:
- Location (city/state)
- Approximate home value
- Type of coverage needed
- Any specific concerns (flood, earthquake, etc.)

IMPORTANT: Never pressure users for personal information. Only ask if they seem genuinely interested.
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
# Helper Functions
# ──────────────────────────────────────────────────────────────

def check_admin_password(password: str) -> bool:
    """Verify admin password"""
    if not ADMIN_PASSWORD_HASH:
        # Default password for demo: "admin123"
        return password == "admin123"
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == ADMIN_PASSWORD_HASH

def extract_contact_info(text: str) -> Dict:
    """Extract email/phone from text"""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    phone_pattern = r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b'
    
    emails = re.findall(email_pattern, text)
    phones = re.findall(phone_pattern, text)
    
    return {
        'email': emails[0] if emails else None,
        'phone': phones[0] if phones else None
    }

def analyze_conversation_for_lead(messages: List[Dict]) -> Dict:
    """Analyze conversation for lead information"""
    full_convo = " ".join([m['content'] for m in messages[-6:]])
    
    lead = {
        'timestamp': datetime.now().isoformat(),
        'session_id': st.session_state.session_id,
        'name': None,
        'email': None,
        'phone': None,
        'location': None,
        'home_value_range': None,
        'interest_level': 'low',
        'conversation_summary': full_convo[:500]  # First 500 chars
    }
    
    # Extract contact info
    contact = extract_contact_info(full_convo)
    lead.update(contact)
    
    # Simple name extraction
    for msg in messages:
        content = msg['content'].lower()
        if "name is" in content:
            parts = content.split("name is")
            if len(parts) > 1:
                potential_name = parts[1].split()[0].strip(".,")
                if len(potential_name) > 1:
                    lead['name'] = potential_name.title()
    
    # Determine interest level
    interest_keywords = ["quote", "price", "buy", "apply", "coverage", "policy"]
    high_interest_keywords = ["email me", "call me", "contact me", "send", "quote now"]
    
    interest_count = sum(1 for kw in interest_keywords if kw in full_convo.lower())
    if any(kw in full_convo.lower() for kw in high_interest_keywords):
        lead['interest_level'] = 'high'
    elif interest_count >= 2:
        lead['interest_level'] = 'medium'
    
    return lead

def estimate_risk_factor(location: str) -> Dict:
    """Simple risk estimation"""
    if not location:
        return {"risk_level": "Standard", "multiplier": 1.0}
    
    loc = location.lower()
    if any(x in loc for x in ["florida", "louisiana", "texas coast", "hurricane"]):
        return {"risk_level": "High", "multiplier": 2.0}
    elif any(x in loc for x in ["california", "wildfire", "earthquake"]):
        return {"risk_level": "Elevated", "multiplier": 1.5}
    else:
        return {"risk_level": "Standard", "multiplier": 1.0}

# ──────────────────────────────────────────────────────────────
# Admin Dashboard Functions
# ──────────────────────────────────────────────────────────────

def show_admin_dashboard():
    """Display admin dashboard"""
    st.title("🔒 Admin Dashboard")
    st.markdown(f"**Logged in as:** {ADMIN_USERNAME}")
    st.markdown(f"**Last login:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    tab1, tab2, tab3 = st.tabs(["📊 Leads", "📈 Analytics", "⚙️ Settings"])
    
    with tab1:
        st.header("Lead Management")
        
        df = get_all_leads()
        if len(df) > 0:
            st.metric("Total Leads", len(df))
            
            # Filter options
            col1, col2 = st.columns(2)
            with col1:
                interest_filter = st.selectbox(
                    "Filter by Interest",
                    ["All", "High", "Medium", "Low"]
                )
            
            with col2:
                date_filter = st.date_input(
                    "Filter by Date",
                    value=None
                )
            
            # Apply filters
            if interest_filter != "All":
                df = df[df['interest_level'] == interest_filter.lower()]
            
            if date_filter:
                df = df[df['timestamp'].str.contains(str(date_filter))]
            
            # Display leads
            st.dataframe(
                df.drop(['ip_address', 'user_agent'], axis=1, errors='ignore'),
                use_container_width=True
            )
            
            # Export options
            csv = df.to_csv(index=False)
            st.download_button(
                "📥 Download CSV",
                csv,
                f"leads_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )
            
            # Delete options
            with st.expander("🗑️ Delete Leads"):
                st.warning("This action cannot be undone!")
                if st.button("Delete ALL Leads", type="secondary"):
                    conn = sqlite3.connect(DB_FILE)
                    c = conn.cursor()
                    c.execute("DELETE FROM leads")
                    conn.commit()
                    conn.close()
                    log_admin_action("delete_all_leads", "All leads deleted")
                    st.success("All leads deleted")
                    st.rerun()
        else:
            st.info("No leads captured yet.")
    
    with tab2:
        st.header("Analytics")
        
        df = get_all_leads()
        if len(df) > 0:
            col1, col2, col3 = st.columns(3)
            with col1:
                high_interest = len(df[df['interest_level'] == 'high'])
                st.metric("High Interest Leads", high_interest)
            
            with col2:
                with_email = df['email'].notna().sum()
                st.metric("Leads with Email", with_email)
            
            with col3:
                today_count = df[df['timestamp'].str.contains(datetime.now().strftime('%Y-%m-%d'))].shape[0]
                st.metric("Today's Leads", today_count)
            
            # Interest level distribution
            st.subheader("Interest Level Distribution")
            interest_counts = df['interest_level'].value_counts()
            st.bar_chart(interest_counts)
            
            # Recent leads timeline
            st.subheader("Recent Leads Timeline")
            df['date'] = pd.to_datetime(df['timestamp']).dt.date
            daily_counts = df.groupby('date').size()
            st.line_chart(daily_counts)
        else:
            st.info("No data available for analytics.")
    
    with tab3:
        st.header("Settings")
        
        st.subheader("Database Info")
        st.info(f"Database file: {DB_FILE}")
        st.info(f"File size: {os.path.getsize(DB_FILE) / 1024:.1f} KB" if os.path.exists(DB_FILE) else "File not found")
        
        st.subheader("Admin Actions")
        if st.button("Backup Database"):
            if os.path.exists(DB_FILE):
                with open(DB_FILE, "rb") as f:
                    st.download_button(
                        "Download Backup",
                        f,
                        f"backup_{datetime.now().strftime('%Y%m%d')}.db",
                        "application/x-sqlite3"
                    )
            log_admin_action("database_backup")
        
        st.subheader("Admin Logs")
        if os.path.exists(DB_FILE):
            conn = sqlite3.connect(DB_FILE)
            logs_df = pd.read_sql_query("SELECT * FROM admin_logs ORDER BY timestamp DESC LIMIT 50", conn)
            conn.close()
            if len(logs_df) > 0:
                st.dataframe(logs_df, use_container_width=True)
            else:
                st.info("No admin logs yet.")
    
    # Logout button
    if st.sidebar.button("🚪 Logout", type="primary"):
        st.session_state.admin_logged_in = False
        log_admin_action("logout")
        st.rerun()

# ──────────────────────────────────────────────────────────────
# Streamlit UI Configuration
# ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Home Insurance Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .stButton>button {
        border-radius: 0.5rem;
        font-weight: 500;
    }
    .admin-badge {
        background-color: #dc3545;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Sidebar (Visible to All Users)
# ──────────────────────────────────────────────────────────────

with st.sidebar:
    # Logo and title
    st.markdown("""
    <div style="text-align: center;">
        <h1 style="font-size: 1.8rem;">🏠</h1>
        <h2>Insurance Assistant</h2>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    # Admin Login Section (Collapsible)
    with st.expander("🔒 Admin Access", expanded=False):
        if not st.session_state.admin_logged_in:
            admin_password = st.text_input(
                "Admin Password", 
                type="password",
                help="Enter admin password to access dashboard"
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Login", use_container_width=True):
                    if admin_password and check_admin_password(admin_password):
                        st.session_state.admin_logged_in = True
                        log_admin_action("login")
                        st.success("Admin login successful!")
                        st.rerun()
                    elif admin_password:
                        st.error("Incorrect password")
            with col2:
                if st.button("Reset", use_container_width=True, type="secondary"):
                    st.rerun()
        else:
            st.success("✅ Admin logged in")
            if st.button("Go to Dashboard", use_container_width=True):
                st.session_state.show_admin = True
                st.rerun()
    
    st.divider()
    
    # User Actions
    st.header("Quick Actions")
    
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    if st.button("📝 Quick Quote Form", use_container_width=True):
        # Store in session to show form in main area
        st.session_state.show_quick_form = True
        st.rerun()
    
    st.divider()
    
    # Information
    st.caption("""
    **About this assistant:**
    - Provides insurance information
    - Helps compare coverage options
    - Guides you through quotes
    - Your data is private and secure
    """)
    
    # Privacy link
    st.markdown("[Privacy Policy](#)")

# ──────────────────────────────────────────────────────────────
# Main Content Area
# ──────────────────────────────────────────────────────────────

# Check if admin is viewing dashboard
if st.session_state.get("show_admin", False) and st.session_state.admin_logged_in:
    show_admin_dashboard()
    st.stop()

# Main Chat Interface
st.markdown("""
<div class="main-header">
    <h1>Home Insurance Assistant</h1>
    <p>Get answers, compare coverage, and explore your options</p>
</div>
""", unsafe_allow_html=True)

# Show quick quote form if requested
if st.session_state.get("show_quick_form", False):
    st.subheader("📋 Quick Quote Request")
    
    with st.form("quick_quote_form"):
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Your Name")
            email = st.text_input("Email Address")
        with col2:
            location = st.text_input("City, State")
            home_value = st.selectbox(
                "Home Value Range",
                ["Under $200k", "$200k-$500k", "$500k-$1M", "Over $1M", "Not sure"]
            )
        
        submitted = st.form_submit_button("Get Free Quotes")
        
        if submitted:
            if email and "@" in email:
                lead_data = {
                    'name': name,
                    'email': email,
                    'phone': None,
                    'location': location,
                    'home_value_range': home_value,
                    'interest_level': 'high',
                    'conversation_summary': 'Quick form submission'
                }
                save_lead_to_db(lead_data)
                st.session_state.user_leads.append(lead_data)
                st.success("✅ Thank you! We'll contact you with personalized quotes.")
                st.session_state.show_quick_form = False
                st.rerun()
            else:
                st.error("Please enter a valid email address")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if user_input := st.chat_input("Type your question about insurance..."):
    user_input = user_input.strip()
    
    # Input validation
    if len(user_input) < 2:
        st.warning("Please type a longer message")
    elif len(user_input) > 800:
        st.warning("Message is too long. Please keep it under 800 characters.")
    elif re.search(r'\b(fuck|shit|damn|asshole|bitch)\b', user_input.lower()):
        st.warning("Please keep the conversation professional.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = chain.invoke(
                        {"input": user_input},
                        config={"configurable": {"session_id": st.session_state.session_id}}
                    )
                    
                    reply = response.content if hasattr(response, "content") else str(response)
                    
                    # Check for contact info and save lead
                    contact_info = extract_contact_info(user_input)
                    if contact_info['email'] or contact_info['phone']:
                        st.success("✅ Contact info received! We'll follow up with more information.")
                        
                        # Analyze conversation and save lead
                        lead_data = analyze_conversation_for_lead(st.session_state.messages)
                        lead_data.update(contact_info)
                        save_lead_to_db(lead_data)
                        st.session_state.user_leads.append(lead_data)
                    
                    # Enhance quote responses
                    quote_triggers = ["quote", "price", "how much", "cost", "rate", "premium"]
                    if any(trigger in user_input.lower() for trigger in quote_triggers):
                        if not (contact_info['email'] or contact_info['phone']):
                            reply += "\n\n**💡 Want personalized quotes?** Share your email for quotes from our partner carriers."
                    
                    st.markdown(reply)
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    
                except Exception as e:
                    logger.error(f"Chat error: {str(e)}")
                    error_msg = "I apologize for the technical issue. Please try again or use the quick quote form."
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Footer
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔒 **Secure & Private**")
    st.caption("Your data is protected")
with col2:
    st.caption("📞 **Live Agent Support**")
    st.caption("Available 9am-6pm EST")
with col3:
    if st.session_state.admin_logged_in:
        st.caption(f"👑 **Admin Mode**")
        if st.button("View Dashboard", type="secondary"):
            st.session_state.show_admin = True
            st.rerun()
    else:
        st.caption("🏠 **Home Insurance**")
        st.caption("Since 2024")

# Hidden admin status indicator (only visible to admin)
if st.session_state.admin_logged_in:
    st.sidebar.markdown(f'<span class="admin-badge">ADMIN</span>', unsafe_allow_html=True)
