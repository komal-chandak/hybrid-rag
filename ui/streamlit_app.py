import streamlit as st
import requests
import pandas as pd
import uuid

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/ask"

st.set_page_config(page_title="Internal Knowledge Assistant", layout="wide")

# ---------------- SESSION STATE INIT ----------------
if "token" not in st.session_state:
    res = requests.post(f"{BASE_URL}/token", timeout=5)
    st.session_state.token = res.json()["access_token"]

# if "user_id" not in st.session_state:
#     st.session_state.user_id = str(uuid.uuid4())

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

if "sessions" not in st.session_state:
    st.session_state.sessions = []

if "preview_image" not in st.session_state:
    st.session_state.preview_image = None

# ---------------- LOAD SESSIONS ----------------
def load_sessions():
    try:
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        res = requests.get(
            f"{BASE_URL}/sessions",
            headers=headers,
            timeout=30
        )
        st.session_state.sessions = res.json()
    except Exception as e:
        st.session_state.sessions = []
        st.warning(f"Failed to load sessions: {str(e)}")

if not st.session_state.sessions:
    load_sessions()

# ---------------- HELPER FUNCTION ----------------

def make_request(method, url, **kwargs):
    def _do_request(token):
        headers = {"Authorization": f"Bearer {token}"}
        return requests.request(method, url, headers=headers, **kwargs)

    res = _do_request(st.session_state.token)

    if res.status_code == 401:
        st.info("Session expired. Refreshing...")
        new_token = requests.post(f"{BASE_URL}/token", timeout=30).json()["access_token"]
        st.session_state.token = new_token
        res = _do_request(new_token)

    return res

# ---------------- SIDEBAR ----------------
# st.sidebar.header("Options")
# show_sources = st.sidebar.toggle("Show Sources", True)
# show_images = st.sidebar.toggle("Show Images", True)
show_sources = True
show_images = True

# New Chat
if st.sidebar.button("➕ New Chat"):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    load_sessions()
    st.rerun()
 
# Session list
st.sidebar.markdown("### Chats")
for s in st.session_state.sessions:
    label = s.get("title") or s["session_id"][:8]

    col1, col2 = st.sidebar.columns([4, 1])

    # ---- OPEN CHAT ----
    with col1:
        if st.button(label, key=f"open_{s['session_id']}"):
            st.session_state.session_id = s["session_id"]

            try:
                history = make_request(
                    "DELETE",
                    f"{BASE_URL}/clear_history/{s['session_id']}"
                ).json()

                st.session_state.messages = history
            except:
                st.session_state.messages = []

            st.rerun()

    # ---- DELETE CHAT ----
    with col2:
        if st.button("🗑️", key=f"delete_{s['session_id']}"):
            st.session_state.confirm_delete = s["session_id"]
            st.rerun()

# ---------------- DELETE CONFIRMATION ----------------
if "confirm_delete" in st.session_state:
    sid = st.session_state.confirm_delete

    st.sidebar.warning("Delete this chat?")

    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("Yes, delete"):
            make_request("DELETE", f"{BASE_URL}/clear_history/{sid}")

            if st.session_state.session_id == sid:
                st.session_state.session_id = str(uuid.uuid4())
                st.session_state.messages = []

            # ✅ FORCE refresh sessions
            st.session_state.sessions = []
            load_sessions()

            del st.session_state.confirm_delete
            st.rerun()

    with col2:
        if st.button("Cancel"):
            del st.session_state.confirm_delete
            st.rerun()


# Debug info 
# st.sidebar.markdown("---")
# st.sidebar.caption(f"User: {st.session_state.user_id[:8]}")
# st.sidebar.caption(f"Session: {st.session_state.session_id[:8]}")

# ---------------- STYLES ----------------
st.markdown("""
<style>
.image-card {
    border: 1px solid #e6e6e6;
    border-radius: 8px;
    padding: 10px;
    height: 220px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.image-box {
    height: 140px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    background: #f6f7f9;
    border-radius: 6px;
}
.image-box img {
    max-height: 140px;
    object-fit: contain;
}
</style>
""", unsafe_allow_html=True)

# ---------------- IMAGE PREVIEW ----------------
@st.dialog("Image Preview")
def show_preview():
    if st.session_state.preview_image:
        st.image(st.session_state.preview_image, width="stretch")

# ---------------- IMAGE GRID ----------------
def render_images(images, msg_index):
    cols = st.columns(6, gap="small")

    for idx, img in enumerate(images):
        with cols[idx % 6]:
            st.image(img, use_container_width=True)

            if st.button("View", key=f"img_{msg_index}_{idx}"):
                st.session_state.preview_image = img
                show_preview()

# ---------------- MESSAGE RENDER ----------------
def render_message(i, msg):
    with st.chat_message(msg["role"]):

        if msg["role"] == "user":
            st.write(msg["content"])
            return

        # Assistant message
        st.write(msg["content"])

        if show_images and msg.get("images"):
            st.caption("Attachments")
            render_images(msg["images"], i)

        if msg.get("tables"):
            for table in msg["tables"]:
                df = pd.DataFrame(table)
                st.dataframe(df)

        if show_sources and msg.get("citations"):
            with st.expander("Sources"):
                for c in msg["citations"]:
                    st.write(f"- {c}")

        # # Retry button (after every message)
        # if st.button("🔄", key=f"retry_{i}"):
        #     st.session_state.retry_query = msg.get("query")
        #     st.session_state.messages.pop(i)
        #     st.rerun()

# ---------------- RENDER CHAT ----------------
for i, msg in enumerate(st.session_state.messages):
    render_message(i, msg)


# ---------------- RETRY BUTTON ----------------
if st.session_state.messages:
    last_msg = st.session_state.messages[-1]

    if last_msg["role"] == "assistant":
        if st.button("🔄"):

            # remove assistant message
            st.session_state.messages.pop()

            # get previous user query safely
            if st.session_state.messages:
                last_user_query = st.session_state.messages[-1]["query"]
                st.session_state.retry_query = last_user_query

            st.rerun()

# ---------------- CHAT INPUT ----------------
prompt = st.chat_input("Ask a question")

# Retry handling
if "retry_query" in st.session_state:
    prompt = st.session_state.retry_query
    force_refresh = True
    del st.session_state.retry_query
else:
    force_refresh = False

# ---------------- HANDLE QUERY ----------------
if prompt:
    if not force_refresh:
        st.session_state.messages.append({
            "role": "user",
            "content": prompt,
            "query": prompt
        })

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            try:
                response = make_request(
                    "POST",
                    API_URL,
                    json={
                        "query": prompt,
                        "force_refresh": force_refresh,
                        "session_id": st.session_state.session_id
                    },
                    timeout=60
                ).json()
            except Exception as e:
                response = {"answer": f"Error{str(e)}."}

    assistant_message = {
        "role": "assistant",
        "content": response.get("answer", ""),
        "images": response.get("images", []),
        "tables": response.get("tables", []),
        "citations": response.get("citations", []),
        "query": prompt
    }

    st.session_state.messages.append(assistant_message)
    st.rerun()