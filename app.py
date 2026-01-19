"""
Q.Prescription - AI-Powered Prescription Analysis
Intelligent prescription processing with OCR and LLM Vision
"""
import streamlit as st
import tempfile
import json
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="Q.Prescription",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Clean CSS with light theme
st.markdown("""
<style>
    @import url('https://rsms.me/inter/inter.css');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hide defaults */
    #MainMenu, footer, header {visibility: hidden;}

    /* Light background */
    .main {
        background: #ffffff;
        padding: 0;
    }

    .block-container {
        padding: 1.5rem 2rem 2rem 2rem;
        max-width: 1400px;
    }

    /* Header */
    .app-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1.5rem 0;
        margin-bottom: 2rem;
        border-bottom: 1px solid #e5e7eb;
    }

    .app-header h1 {
        font-size: 32px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -1px;
    }

    .app-header .tagline {
        font-size: 14px;
        color: #6b7280;
        margin-top: 4px;
        font-weight: 400;
    }

    .app-header .quick-stats {
        display: flex;
        gap: 3rem;
        font-size: 15px;
        color: #6b7280;
        font-weight: 500;
    }

    .app-header .quick-stats .stat-item {
        text-align: center;
    }

    .app-header .quick-stats .stat-num {
        font-size: 24px;
        font-weight: 700;
        color: #1f2937;
        display: block;
        margin-bottom: 2px;
    }

    .app-header .quick-stats .stat-label {
        font-size: 11px;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Hero */
    .hero {
        text-align: center;
        padding: 4rem 2rem;
        background: #f9fafb;
        border-radius: 16px;
        border: 2px dashed #d1d5db;
        margin: 2rem 0;
    }

    .hero h1 {
        font-size: 56px;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        letter-spacing: -2px;
    }

    .hero .tagline {
        font-size: 20px;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #f9fafb;
        border-right: 1px solid #e5e7eb;
        min-width: 320px !important;
        max-width: 320px !important;
    }

    section[data-testid="stSidebar"] > div {
        padding: 1rem;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 13px;
        font-weight: 700;
        color: #6b7280;
        margin: 1rem 0 0.75rem 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    section[data-testid="stSidebar"] hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1rem 0;
    }

    /* Sidebar expanders */
    section[data-testid="stSidebar"] .streamlit-expanderHeader {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 6px;
        font-size: 12px;
        padding: 0.5rem;
        margin-bottom: 0.5rem;
    }

    section[data-testid="stSidebar"] .streamlit-expanderHeader:hover {
        border-color: #667eea;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] .stButton > button {
        font-size: 13px;
        padding: 0.5rem;
    }

    /* Sidebar metrics */
    section[data-testid="stSidebar"] .stMetric {
        background: #ffffff;
        padding: 0.75rem;
        border-radius: 6px;
        border: 1px solid #e5e7eb;
    }

    section[data-testid="stSidebar"] .stMetric label {
        font-size: 11px !important;
        color: #6b7280 !important;
    }

    section[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
        font-size: 20px !important;
        color: #1f2937 !important;
    }

    /* Force sidebar to always show */
    [data-testid="collapsedControl"] {
        display: block !important;
    }

    /* Sidebar toggle button */
    button[kind="header"] {
        color: #1f2937 !important;
    }

    /* Cards */
    .card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        transition: all 0.2s;
    }

    .card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: transparent;
        border-bottom: 1px solid #e5e7eb;
        padding: 0;
        margin-bottom: 2rem;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 0.75rem 1.5rem;
        color: #6b7280;
        border: none;
        border-bottom: 2px solid transparent;
        font-weight: 500;
        font-size: 14px;
        background: transparent;
    }

    .stTabs [aria-selected="true"] {
        color: #667eea;
        border-bottom-color: #667eea;
        background: transparent;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.625rem 1.25rem;
        font-weight: 600;
        font-size: 14px;
        transition: all 0.2s;
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }

    /* Stats */
    .stat-card {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.2s;
    }

    .stat-card:hover {
        border-color: #667eea;
    }

    .stat-num {
        font-size: 36px;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }

    .stat-label {
        font-size: 12px;
        color: #6b7280;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 600;
        display: block;
        margin-bottom: 0.5rem;
    }

    .stat-indicator {
        font-size: 13px;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    .stat-good { color: #16a34a; }
    .stat-warning { color: #d97706; }
    .stat-bad { color: #dc2626; }

    /* Document Item */
    .doc-item {
        background: #ffffff;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 1.25rem;
        margin-bottom: 1rem;
        transition: all 0.2s;
        display: flex;
        gap: 1rem;
    }

    .doc-item:hover {
        border-color: #667eea;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.15);
    }

    .doc-preview {
        width: 80px;
        height: 100px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 32px;
        flex-shrink: 0;
    }

    .doc-content {
        flex: 1;
        min-width: 0;
    }

    .doc-name {
        font-size: 16px;
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.75rem;
        word-break: break-word;
    }

    .doc-meta {
        display: flex;
        gap: 0.75rem;
        align-items: center;
        flex-wrap: wrap;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        border-radius: 6px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.3px;
    }

    .badge-success {
        background: rgba(22, 163, 74, 0.1);
        color: #16a34a;
        border: 1px solid #16a34a;
    }

    .badge-warning {
        background: rgba(217, 119, 6, 0.1);
        color: #d97706;
        border: 1px solid #d97706;
    }

    .badge-danger {
        background: rgba(220, 38, 38, 0.1);
        color: #dc2626;
        border: 1px solid #dc2626;
    }

    .badge-info {
        background: rgba(102, 126, 234, 0.1);
        color: #667eea;
        border: 1px solid #667eea;
    }

    .badge-secondary {
        background: rgba(107, 114, 128, 0.1);
        color: #6b7280;
        border: 1px solid #d1d5db;
    }

    /* Status dot */
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
    }

    .status-success { background: #16a34a; }
    .status-processing {
        background: #d97706;
        animation: pulse 1.5s infinite;
    }
    .status-failed { background: #dc2626; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }

    /* Select */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        background: #ffffff;
        color: #1f2937;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        font-size: 14px;
        font-weight: 600;
        padding: 0.75rem 1rem;
        color: #1f2937;
    }

    .streamlit-expanderHeader:hover {
        border-color: #667eea;
    }

    /* Tabs in expander */
    .streamlit-expanderContent .stTabs [data-baseweb="tab-list"] {
        background: #f9fafb;
        border-bottom: 1px solid #e5e7eb;
        padding: 0.5rem;
        gap: 0.5rem;
        border-radius: 8px 8px 0 0;
        margin-bottom: 0;
    }

    .streamlit-expanderContent .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-size: 13px;
        background: transparent;
        border-radius: 6px;
    }

    .streamlit-expanderContent .stTabs [aria-selected="true"] {
        background: #667eea;
        color: white;
        border-bottom: none;
    }

    /* Images */
    .streamlit-expanderContent img {
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin: 1rem 0;
    }

    /* Text area */
    .stTextArea textarea {
        background: #f9fafb !important;
        color: #1f2937 !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
        font-family: 'Monaco', 'Menlo', monospace !important;
        font-size: 12px !important;
    }

    /* JSON viewer */
    .stJson {
        background: #f9fafb !important;
        border: 1px solid #e5e7eb !important;
        border-radius: 8px !important;
    }

    /* Alerts */
    .stAlert {
        border-radius: 8px;
        border: none;
        font-size: 14px;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        color: #1f2937;
    }

    /* Text colors */
    .text-muted {
        color: #6b7280;
        font-size: 13px;
    }

    /* Metric labels */
    .stMetric label {
        color: #6b7280 !important;
        font-size: 12px !important;
    }

    .stMetric [data-testid="stMetricValue"] {
        color: #1f2937 !important;
        font-size: 24px !important;
    }

    /* Delete button style */
    .delete-btn {
        background: rgba(220, 38, 38, 0.1) !important;
        color: #dc2626 !important;
        border: 1px solid #dc2626 !important;
    }

    .delete-btn:hover {
        background: rgba(220, 38, 38, 0.2) !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.2) !important;
    }
</style>
""", unsafe_allow_html=True)

from core.config import Config
from extraction.prescription_processor import prescription_processor

# ========================= SESSION STATE =========================
if "processed_prescriptions" not in st.session_state:
    st.session_state.processed_prescriptions = []

# ========================= HELPERS =========================
def get_prescription_type_badge(rx_type):
    """Get badge class for prescription type"""
    badges = {
        "handwritten": ("✍️ Handwritten", "warning"),
        "printed": ("🖨️ Printed", "success"),
        "mixed": ("📝 Mixed", "info"),
        "digital": ("💻 Digital", "secondary")
    }
    return badges.get(rx_type, ("❓ Unknown", "secondary"))


def get_prescription_stats():
    """Get statistics about processed prescriptions"""
    prescriptions = st.session_state.processed_prescriptions
    if not prescriptions:
        return {"total": 0, "with_signature": 0, "total_meds": 0, "llm_enhanced": 0}

    total = len(prescriptions)
    with_signature = sum(1 for p in prescriptions
                        if p["prescription"].doctor_signature
                        and p["prescription"].doctor_signature.is_present)
    total_meds = sum(len(p["prescription"].medications) for p in prescriptions)
    llm_enhanced = sum(1 for p in prescriptions if p["prescription"].llm_enhanced)

    return {
        "total": total,
        "with_signature": with_signature,
        "total_meds": total_meds,
        "llm_enhanced": llm_enhanced
    }


# ========================= SIDEBAR =========================
def render_sidebar():
    """Render sidebar with upload and stats"""
    with st.sidebar:
        # Header
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #667eea; font-size: 24px; margin: 0;">💊 Q.Prescription</h2>
            <p style="color: #888; font-size: 12px; margin: 0.5rem 0 0 0;">AI Prescription Analyzer</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # Quick Upload Section
        st.markdown("### 📤 Quick Upload")
        files = st.file_uploader(
            "Drop prescriptions here",
            type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
            label_visibility="collapsed",
            key="sidebar_upload"
        )

        # Limit the number of files to 10
        if files and len(files) > 10:
            st.warning("You can only upload up to 10 files at a time.")
            files = files[:10]

        if files:
            force_vision = st.checkbox("Force LLM Vision", value=True, key="sidebar_force_vision")

            if st.button("🔍 Analyze", key="sidebar_analyze", use_container_width=True):
                progress = st.progress(0)
                status_text = st.empty()

                for i, f in enumerate(files):
                    status_text.text(f"{i+1}/{len(files)}: {f.name[:20]}...")

                    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                        tmp.write(f.getbuffer())
                        tmp_path = tmp.name

                    try:
                        prescription, metadata = prescription_processor.process(tmp_path, force_vision=force_vision)
                        st.session_state.processed_prescriptions.append({
                            "filename": f.name,
                            "prescription": prescription,
                            "metadata": metadata,
                            "image_data": f.getvalue(),
                            "processed_at": datetime.now()
                        })
                    except Exception as e:
                        st.error(f"Error: {e}")
                    finally:
                        try:
                            Path(tmp_path).unlink()
                        except:
                            pass

                    progress.progress((i + 1) / len(files))

                progress.empty()
                status_text.empty()
                st.success("✓ Processed!")
                st.rerun()

        st.markdown("---")

        # Stats Section
        #st.markdown("### 📊 Stats")
        #stats = get_prescription_stats()

        #col1, col2 = st.columns(2)
        #with col1:
           # st.metric("Processed", stats["total"])
        #with col2:
           # st.metric("Signatures", stats["with_signature"])

        #col1, col2 = st.columns(2)
        #with col1:
           # st.metric("Medications", stats["total_meds"])
        #with col2:
            #st.metric("LLM Enhanced", stats["llm_enhanced"])

        #st.markdown("---")

        # About
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div style="font-size: 11px; color: #666;">
        <b>Q.Prescription v1.0</b><b>
        AI-powered prescription analysis<b>
        <br>
        Features:<br>
        • OCR + LLM Vision extraction<br>
        • Handwriting recognition<br>
        • Signature detection<br>
        • Medication parsing
        </div>
        """, unsafe_allow_html=True)


# ========================= MAIN CONTENT =========================
def render_upload_tab():
    """Render the upload/process tab"""
    st.markdown("""
    <div style="background: #f0f4ff; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #667eea;">
        <p style="margin: 0; color: #374151; font-size: 14px;">
            <strong>AI-Powered Prescription Analysis</strong><br>
            <strong>High confidence:</strong> OCR → LLM text structuring (fast & cheap)<br>
            <strong>Low confidence:</strong> OCR → LLM Vision (for handwritten text)<br>
            <strong>Signatures:</strong> Always detected with LLM Vision
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Upload section
    col1, col2 = st.columns([3, 1])

    with col1:
        uploaded_files = st.file_uploader(
            "Upload prescriptions",
            type=["png", "jpg", "jpeg"],
            key="main_prescription_upload",
            accept_multiple_files=True,
            help="Supports handwritten, printed, and mixed prescriptions"
        )

    with col2:
        st.markdown("**Options**")
        force_vision = st.checkbox(
            "Force LLM Vision",
            value=False,
            help="Always use LLM vision even for clear printed text. When off, LLM is only used for low-confidence OCR and signatures.",
            key="main_force_vision"
        )

    if uploaded_files:
        st.markdown(f"**{len(uploaded_files)} file(s) selected**")

        # Preview uploaded files
        preview_cols = st.columns(min(len(uploaded_files), 4))
        for i, f in enumerate(uploaded_files[:4]):
            with preview_cols[i]:
                st.image(f, caption=f.name[:20], width=150)

        if len(uploaded_files) > 4:
            st.caption(f"... and {len(uploaded_files) - 4} more")

        if st.button("🔍 Analyze All Prescriptions", key="analyze_rx_batch", use_container_width=True):
            progress = st.progress(0)
            status_text = st.empty()

            for idx, uploaded_file in enumerate(uploaded_files):
                status_text.text(f"Processing {idx + 1}/{len(uploaded_files)}: {uploaded_file.name}")

                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getbuffer())
                    tmp_path = tmp.name

                try:
                    prescription, metadata = prescription_processor.process(tmp_path, force_vision=force_vision)
                    st.session_state.processed_prescriptions.append({
                        "filename": uploaded_file.name,
                        "prescription": prescription,
                        "metadata": metadata,
                        "image_data": uploaded_file.getvalue(),
                        "processed_at": datetime.now()
                    })
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                finally:
                    try:
                        Path(tmp_path).unlink()
                    except:
                        pass

                progress.progress((idx + 1) / len(uploaded_files))

            progress.empty()
            status_text.empty()
            st.success(f"✓ Processed {len(uploaded_files)} prescriptions!")
            st.rerun()


def render_library_tab():
    """Render the prescription library tab"""
    prescriptions = st.session_state.processed_prescriptions

    if not prescriptions:
        st.markdown("""
        <div style="text-align: center; padding: 3rem; background: #222; border-radius: 12px; border: 2px dashed #444;">
            <h3 style="color: #888;">No prescriptions processed yet</h3>
            <p style="color: #666;">Upload prescriptions in the "Process New" tab to see them here.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Filter and sort controls
    col1, col2, col3 = st.columns([2, 2, 1])

    with col1:
        # Get unique types, handling enum values
        type_set = set()
        for p in prescriptions:
            rx_type = p["prescription"].prescription_type
            if rx_type:
                type_set.add(rx_type.value if hasattr(rx_type, 'value') else str(rx_type))
            else:
                type_set.add("unknown")
        type_options = ["All"] + sorted(list(type_set))
        filter_type = st.selectbox("Filter by type", type_options, key="rx_lib_filter")

    with col2:
        sort_options = ["Newest first", "Oldest first", "Name A-Z", "Most medications"]
        sort_by = st.selectbox("Sort by", sort_options, key="rx_lib_sort")

    with col3:
        st.metric("Total", len(prescriptions))

    # Filter
    filtered = prescriptions
    if filter_type != "All":
        filtered = [p for p in prescriptions
                   if (p["prescription"].prescription_type.value
                       if hasattr(p["prescription"].prescription_type, 'value')
                       else str(p["prescription"].prescription_type or "unknown")) == filter_type]

    # Sort
    if sort_by == "Newest first":
        filtered.sort(key=lambda x: x["processed_at"], reverse=True)
    elif sort_by == "Oldest first":
        filtered.sort(key=lambda x: x["processed_at"])
    elif sort_by == "Name A-Z":
        filtered.sort(key=lambda x: x["filename"])
    else:
        filtered.sort(key=lambda x: len(x["prescription"].medications), reverse=True)

    st.markdown(f'<div class="text-muted mb-2">Showing {len(filtered)} prescriptions</div>', unsafe_allow_html=True)

    # Display prescriptions
    for idx, item in enumerate(filtered):
        rx = item["prescription"]
        meta = item["metadata"]

        with st.container():
            col_preview, col_content, col_actions = st.columns([1, 6, 1])

            with col_preview:
                st.markdown('<div class="doc-preview">💊</div>', unsafe_allow_html=True)

            with col_content:
                st.markdown(f'<div class="doc-name">{item["filename"]}</div>', unsafe_allow_html=True)

                # Badges - handle enum type
                rx_type = rx.prescription_type
                if hasattr(rx_type, 'value'):
                    rx_type_str = rx_type.value
                else:
                    rx_type_str = str(rx_type) if rx_type else "unknown"

                type_badge, type_class = get_prescription_type_badge(rx_type_str)
                sig_badge = "✓ Signature" if rx.doctor_signature and rx.doctor_signature.is_present else "✗ No Sig"
                sig_class = "success" if rx.doctor_signature and rx.doctor_signature.is_present else "secondary"
                # Extraction method badge with more detail
                extraction_method = meta.get("extraction_method", "")
                if extraction_method == "ocr_plus_llm_vision":
                    method_badge = "👁️ LLM Vision"
                    method_class = "info"
                elif extraction_method == "ocr_plus_llm_text":
                    method_badge = "🤖 LLM Text"
                    method_class = "warning"
                elif rx.llm_enhanced:
                    method_badge = "🤖 LLM"
                    method_class = "info"
                else:
                    method_badge = "📝 Regex"
                    method_class = "secondary"
                med_count = len(rx.medications)
                date_str = item["processed_at"].strftime("%b %d, %H:%M")

                st.markdown(f"""
                <div class="doc-meta">
                    <span class="badge badge-{type_class}">{type_badge}</span>
                    <span class="badge badge-{sig_class}">{sig_badge}</span>
                    <span class="badge badge-{method_class}">{method_badge}</span>
                    <span class="badge badge-secondary">💊 {med_count} meds</span>
                    <span class="text-muted">{date_str}</span>
                </div>
                """, unsafe_allow_html=True)

            with col_actions:
                if st.button("🗑️", key=f"del_rx_{idx}", help="Delete"):
                    st.session_state.processed_prescriptions.remove(item)
                    st.rerun()

            # Expandable details
            with st.expander("📄 View Details", expanded=False):
                detail_tabs = st.tabs(["Preview", "Patient & Doctor", "Medications", "Signature", "Processing Info"])

                with detail_tabs[0]:
                    if "image_data" in item:
                        st.image(item["image_data"], width=700)
                    else:
                        st.warning("No image data available.")

                with detail_tabs[1]:
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**👤 Patient**")
                        if rx.patient and (rx.patient.name or rx.patient.age or rx.patient.gender):
                            st.text_input("Name", rx.patient.name or "N/A", disabled=True, key=f"pat_name_{idx}")
                            st.text_input("Age", rx.patient.age or "N/A", disabled=True, key=f"pat_age_{idx}")
                            st.text_input("Gender", rx.patient.gender or "N/A", disabled=True, key=f"pat_gender_{idx}")
                            st.text_input("Address", rx.patient.address or "N/A", disabled=True, key=f"pat_addr_{idx}")
                            st.text_input("Phone", rx.patient.phone or "N/A", disabled=True, key=f"pat_phone_{idx}")
                        else:
                            st.info("No patient information available.")

                    with col2:
                        st.markdown("**🩺 Doctor**")
                        if rx.doctor and (rx.doctor.name or rx.doctor.title or rx.doctor.specialty):
                            st.text_input("Name", rx.doctor.name or "N/A", disabled=True, key=f"doc_name_{idx}")
                            st.text_input("Title", rx.doctor.title or "N/A", disabled=True, key=f"doc_title_{idx}")
                            st.text_input("Specialty", rx.doctor.specialty or "N/A", disabled=True, key=f"doc_spec_{idx}")
                            st.text_input("License No.", rx.doctor.license_number or "N/A", disabled=True, key=f"doc_lic_{idx}")
                            st.text_input("Phone", rx.doctor.phone or "N/A", disabled=True, key=f"doc_phone_{idx}")
                        else:
                            st.info("No doctor information available.")

                with detail_tabs[2]:
                    if rx.medications:
                        for i, med in enumerate(rx.medications, 1):
                            st.markdown(f"""
                            <div style="background: #1a1a1a; padding: 1rem; border-radius: 8px; margin-bottom: 0.5rem; border-left: 3px solid #667eea;">
                                <strong style="color: #fff;">{i}. {med.name}</strong>
                            </div>
                            """, unsafe_allow_html=True)
                            med_cols = st.columns(4)
                            with med_cols[0]:
                                st.markdown(f"**Dosage:** {med.dosage or 'N/A'}")
                            with med_cols[1]:
                                st.markdown(f"**Qty:** {med.quantity or 'N/A'}")
                            with med_cols[2]:
                                st.markdown(f"**Freq:** {med.frequency or 'N/A'}")
                            with med_cols[3]:
                                hw = "✍️ Handwritten" if med.is_handwritten else "🖨️ Printed"
                                st.markdown(f"**Source:** {hw}")
                            if med.instructions:
                                st.caption(f"Instructions: {med.instructions}")
                            st.markdown("---")
                    else:
                        st.warning("No medications extracted")

                with detail_tabs[3]:
                    if rx.doctor_signature:
                        sig = rx.doctor_signature
                        if sig.is_present:
                            st.success("✓ Signature detected!")
                            sig_cols = st.columns(4)
                            with sig_cols[0]:
                                st.metric("Signer", sig.signer_name or "Unknown")
                            with sig_cols[1]:
                                st.metric("Title", sig.signer_title or "N/A")
                            with sig_cols[2]:
                                st.metric("Location", sig.location or "N/A")
                            with sig_cols[3]:
                                st.metric("Legible", "Yes" if sig.is_legible else "No")
                            if sig.confidence:
                                st.progress(sig.confidence, text=f"Confidence: {sig.confidence:.1%}")
                        else:
                            st.warning("No signature detected")
                    else:
                        st.info("Signature analysis not performed")

                    # Handwriting analysis
                    if rx.handwriting_analysis and rx.handwriting_analysis.has_handwritten_content:
                        st.markdown("**📝 Handwriting Analysis**")
                        st.info(f"Handwritten sections: {', '.join(rx.handwriting_analysis.handwritten_sections) or 'various'}")
                        if rx.handwriting_analysis.llm_interpretation:
                            st.text_area("LLM Interpretation", rx.handwriting_analysis.llm_interpretation, disabled=True, key=f"hw_interp_{idx}")

                with detail_tabs[4]:
                    # Processing Info tab - shows extraction method details
                    st.markdown("### How was this prescription processed?")

                    # Extraction method explanation - handle 3 cases
                    extraction_method = meta.get("extraction_method", "")

                    if extraction_method == "ocr_plus_llm_vision":
                        st.markdown("""
                        <div style="background: #e0f2fe; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #0284c7;">
                            <strong style="color: #0284c7;">👁️ LLM Vision + OCR</strong>
                            <p style="color: #374151; margin: 0.5rem 0 0 0; font-size: 14px;">
                                The OCR confidence was <strong>below 60%</strong> (likely handwritten or unclear text),
                                so <strong>GPT-4 Vision</strong> analyzed the image directly to extract prescription content.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    elif extraction_method == "ocr_plus_llm_text":
                        st.markdown("""
                        <div style="background: #fef3c7; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #d97706;">
                            <strong style="color: #d97706;">🤖 OCR + LLM Text Structuring</strong>
                            <p style="color: #374151; margin: 0.5rem 0 0 0; font-size: 14px;">
                                The OCR confidence was <strong>above 60%</strong> (clear printed text).
                                OCR extracted the text, then <strong>LLM structured it into JSON</strong> (no vision needed - cheaper & faster).
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: #f3f4f6; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #6b7280;">
                            <strong style="color: #6b7280;">📝 OCR + Regex (Fallback)</strong>
                            <p style="color: #374151; margin: 0.5rem 0 0 0; font-size: 14px;">
                                LLM was not available. OCR extracted text, and <strong>regex patterns</strong> were used to structure it.
                                Results may be less accurate.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)

                    # Confidence metrics
                    st.markdown("**📊 Confidence Scores**")
                    conf_cols = st.columns(3)

                    with conf_cols[0]:
                        ocr_conf = rx.ocr_confidence or 0
                        st.metric("OCR Confidence", f"{ocr_conf:.1%}")
                        st.progress(ocr_conf)

                    with conf_cols[1]:
                        overall_conf = rx.confidence_score or ocr_conf
                        st.metric("Overall Confidence", f"{overall_conf:.1%}")
                        st.progress(overall_conf)

                    with conf_cols[2]:
                        threshold = 0.6
                        status = "✅ Above" if ocr_conf >= threshold else "⚠️ Below"
                        st.metric("vs Threshold (60%)", status)

                    # Processing stages from metadata
                    if meta.get("processing_stages"):
                        st.markdown("**🔄 Processing Stages**")
                        for stage in meta["processing_stages"]:
                            stage_name = stage.get("stage", "unknown")
                            stage_time = stage.get("time", 0)
                            stage_reason = stage.get("reason", "")

                            if stage_name == "ocr":
                                icon = "1️⃣"
                                label = "OCR Text Extraction"
                            elif stage_name == "vision":
                                icon = "2️⃣"
                                label = "LLM Vision Analysis (image → JSON)"
                            elif stage_name == "llm_text_structuring":
                                icon = "2️⃣"
                                label = "LLM Text Structuring (text → JSON)"
                            elif stage_name == "ocr_parsing" or stage_name == "ocr_parsing_regex":
                                icon = "2️⃣"
                                label = "Regex Parsing (fallback)"
                            elif stage_name == "signature_detection":
                                icon = "3️⃣"
                                label = "Signature Detection (LLM Vision)"
                            else:
                                icon = "🔹"
                                label = stage_name

                            st.markdown(f"""
                            <div style="background: #f9fafb; padding: 0.75rem; border-radius: 6px; margin-bottom: 0.5rem; border: 1px solid #e5e7eb;">
                                <span style="color: #1f2937;">{icon} <strong>{label}</strong></span>
                                <span style="float: right; color: #6b7280;">{stage_time:.2f}s</span>
                                {f'<br><span style="color: #6b7280; font-size: 12px;">{stage_reason}</span>' if stage_reason else ''}
                            </div>
                            """, unsafe_allow_html=True)

                    # Total processing time
                    if meta.get("total_processing_time"):
                        st.markdown(f"**Total Processing Time:** {meta['total_processing_time']:.2f} seconds")

                # Display raw OCR text
                if rx.ocr_text:
                    st.markdown("**📄 Raw OCR Text**")
                    st.text_area("OCR Output", rx.ocr_text, height=200, disabled=True, key=f"ocr_text_{idx}")

    # Export and clear buttons
    if filtered:
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📥 Export All as JSON", use_container_width=True):
                export_data = [
                    {
                        "filename": p["filename"],
                        "data": p["prescription"].dict()  # Ensure the prescription data is serialized correctly
                    }
                    for p in filtered
                ]
                st.json(export_data)
        with col2:
            if st.button("🗑️ Clear All", use_container_width=True):
                st.session_state.processed_prescriptions = []
                st.rerun()


# ========================= MAIN =========================
def main():
    # Simple header
    st.markdown("""
    <div style="margin-bottom: 1.5rem;">
        <h1 style="font-size: 28px; font-weight: 700; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0;">💊 Q.Prescription</h1>
        <p style="color: #6b7280; font-size: 14px; margin: 0.25rem 0 0 0;">AI-Powered Prescription Analysis</p>
    </div>
    """, unsafe_allow_html=True)

    # Main tabs
    tab1, tab2 = st.tabs(["📤 Process New", "📚 Library"])

    with tab1:
        render_upload_tab()

    with tab2:
        render_library_tab()


# ========================= ENTRY =========================
if __name__ == "__main__":
    try:
        Config.validate()
        main()
    except Exception as e:
        st.error(f"Configuration error: {e}")
        st.info("Please set your API keys in the .env file")
