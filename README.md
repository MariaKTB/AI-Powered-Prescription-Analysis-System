# Q.Prescription: AI-Powered Prescription Analysis System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Q.Prescription** is an intelligent prescription processing system that combines OCR, LLM text structuring, and LLM Vision to extract structured data from medical prescriptions, including handwritten text and doctor signatures.

---

## Project Overview

### Problem Statement

Medical prescription processing presents unique challenges that differ from standard document processing:

- **Handwritten Content**: Prescriptions often contain handwritten medication names, dosages, and instructions that standard OCR cannot reliably interpret
- **Signature Verification**: Doctor signatures are critical for prescription validity but difficult to detect automatically
- **Mixed Content**: Many prescriptions combine printed headers with handwritten medication details
- **Unstructured Data**: Converting free-form prescription text into structured, actionable data (JSON) for pharmacy systems

### Why Prescriptions Instead of Invoices?

This project initially targeted invoice processing. However, invoices are typically fully printed documents where OCR alone achieves high accuracy, making LLM integration less valuable. We pivoted to medical prescriptions because:

1. **Handwritten text** requires intelligent interpretation beyond OCR
2. **Signature detection** requires visual analysis
3. **LLM adds significant value** in understanding context and structuring data
4. **Results showed measurably better accuracy** with LLM vs regex-only parsing

### Solution

Q.Prescription implements a **three-stage adaptive pipeline**:

1. **Stage 1 - OCR**: PaddleOCR extracts text with multi-factor confidence scoring
2. **Stage 2 - Intelligent Routing**:
   - **High confidence (>60%)**: LLM structures OCR text into JSON (fast, cheap)
   - **Low confidence (<60%)**: LLM Vision analyzes image directly (better for handwriting)
3. **Stage 3 - Signature Detection**: LLM Vision always analyzes signatures

---

## Architecture

### System Components

```
+-------------------------------------------------------------+
|                     Streamlit UI                             |
|  (Upload Interface, Prescription Library, Export)            |
+-----------------------------+-------------------------------+
                              |
+-----------------------------v-------------------------------+
|                 Processing Pipeline                          |
|  +-------------+  +--------------+  +--------------------+   |
|  |    OCR      |  | Confidence   |  |   LLM Routing      |   |
|  | (PaddleOCR) |  |  Calculator  |  | (Vision vs Text)   |   |
|  +-------------+  +--------------+  +--------------------+   |
+-----------------------------+-------------------------------+
                              |
+-----------------------------v-------------------------------+
|                  Extraction Layer                            |
|  +---------------+  +----------------+  +----------------+   |
|  | LLM Vision    |  | LLM Text       |  | Signature      |   |
|  | (GPT-4o)      |  | (GPT-4o-mini)  |  | Detector       |   |
|  +---------------+  +----------------+  +----------------+   |
+-------------------------------------------------------------+
```

### Processing Flow

```
Upload Image
     |
     v
[PDF Conversion] (if needed)
     |
     v
[OCR Extraction] --> Raw Text + Confidence Score
     |
     v
[Confidence Check]
     |
     +-- >= 60% --> [LLM Text Structuring] --> JSON
     |
     +-- < 60%  --> [LLM Vision Analysis] --> JSON
     |
     v
[Signature Detection] (always LLM Vision)
     |
     v
Structured Prescription Data
```

---

## Key Features

### 1. Adaptive OCR-to-LLM Pipeline
- **PaddleOCR** with preprocessing (grayscale, denoising, adaptive thresholding)
- **Multi-factor confidence calculation**: 50% base OCR + 25% text density + 25% content length
- **Intelligent routing** based on confidence threshold (60%)

### 2. Dual LLM Processing Modes
| Mode | When Used | Model | Cost |
|------|-----------|-------|------|
| **LLM Text** | OCR confidence >= 60% | GPT-4o-mini | Low |
| **LLM Vision** | OCR confidence < 60% | GPT-4o | Higher |

### 3. Comprehensive Data Extraction
- **Patient Information**: Name, age, gender, address, phone
- **Doctor Information**: Name, title, specialty, license number
- **Hospital/Clinic Details**: Name, address, contact info
- **Medications**: Name, dosage, quantity, frequency, instructions
- **Signature Analysis**: Presence, legibility, signer name, confidence

### 4. Signature Detection
- Always uses LLM Vision for reliable signature detection
- Identifies signature location, legibility, and signer information
- Reports confidence score for signature presence

### 5. Handwriting Analysis
- Detects handwritten vs printed content
- Identifies which sections are handwritten
- Provides LLM interpretation of handwritten text

---

## Installation

### Prerequisites

- Python 3.10 or higher
- pip package manager
- OpenAI API key (required)

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/q-prescriptions.git
cd q-prescriptions
```

2. **Create virtual environment**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure API keys**
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

5. **Run the application**
```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

---

## Project Structure

```
q-prescriptions/
├── app.py                          # Main Streamlit app
├── requirements.txt                # Dependencies
├── .env.template                   # Environment template
├── README.md                       # Documentation
├── QUICKSTART.md                   # Documentation
├── Technical_report.pdf	        # Documentation
│
├── core/
│   ├── __init__.py
│   └── config.py                   # Configuration
│   └── processor.py                 
│
├── extraction/
│   ├── __init__.py
│   ├── ocr.py                      # PaddleOCR with confidence
│   ├── ocr_parser.py               # Regex fallback
│   ├── llm_extractor.py            # LLM text structuring
│   ├── vision_extractor.py         # GPT-4o Vision
│   ├── prescription_processor.py   # Main pipeline
│   ├── pdf_converter.py            # PDF support
│   └── schema.py                   # Pydantic models
│
└── prescription_dataset/           # Your test images
    ├── handwriting/
    ├── mixed/
    └── printed/

```

---

## Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `OPENAI_API_KEY` | Yes | OpenAI API key for LLM | - |
| `ANTHROPIC_API_KEY` | No | Anthropic API key (fallback) | - |
| `OCR_LANGUAGE` | No | OCR language | en |
| `OCR_USE_GPU` | No | Use GPU for OCR | False |
| `DEFAULT_LLM_MODEL` | No | Default LLM model | gpt-4o-mini |
| `LLM_TEMPERATURE` | No | LLM temperature | 0 |
| `MAX_FILE_SIZE_MB` | No | Maximum file size | 50 |

---

## Data Schema

### ExtractedPrescription

```python
class ExtractedPrescription:
    prescription_type: PrescriptionType  # handwritten, printed, mixed, digital
    prescription_date: Optional[str]
    patient: Optional[PatientInfo]
    doctor: Optional[DoctorInfo]
    hospital: Optional[HospitalInfo]
    medications: List[MedicationItem]
    doctor_signature: Optional[SignatureInfo]
    handwriting_analysis: Optional[HandwritingAnalysis]
    ocr_text: Optional[str]
    ocr_confidence: Optional[float]
    confidence_score: Optional[float]
    llm_enhanced: bool
    extraction_method: Optional[str]
```

### MedicationItem

```python
class MedicationItem:
    name: str
    dosage: Optional[str]
    quantity: Optional[str]
    frequency: Optional[str]
    duration: Optional[str]
    instructions: Optional[str]
    is_handwritten: bool
```

---

## Usage Examples

### Basic Usage

1. Open the application at `http://localhost:8501`
2. Use the main area to upload prescription images
3. Click "Analyze" to process
4. View results in the Library tab

### Processing Options

- **Force LLM Vision**: Always use vision analysis regardless of OCR confidence
- **Default mode**: Let the system decide based on confidence threshold

### Export

- Export processed prescriptions as JSON from the Library tab

---

## Technical Details

### OCR Confidence Calculation

```python
# Multi-factor confidence
combined_confidence = (
    base_ocr_confidence * 0.5 +    # 50% from OCR
    text_density_score * 0.25 +     # 25% from text density
    content_length_score * 0.25     # 25% from content length
)
```

### Confidence Threshold

- **60%** threshold determines LLM routing
- Below 60%: Use LLM Vision (better for handwritten)
- Above 60%: Use LLM Text (cheaper, faster)

---

## Troubleshooting

### Common Issues

**Issue**: "OPENAI_API_KEY not set"
- **Solution**: Create `.env` file with your API key

**Issue**: OCR confidence very low
- **Solution**: Ensure images are high resolution (300+ DPI)

**Issue**: PaddleOCR installation fails
- **Solution**: Install PaddlePaddle first: `pip install paddlepaddle==2.6.2`

**Issue**: Memory issues with large images
- **Solution**: Reduce image resolution before processing

---

## Academic Context

This project was developed as part of a Master's program in AI/Data Science, demonstrating:

- Integration of OCR and LLM technologies
- Adaptive processing pipelines
- Real-world document intelligence challenges
- Production-ready application architecture

### Key Learning Outcomes

1. When LLM adds value vs when simpler solutions suffice
2. Multi-factor confidence scoring for intelligent routing
3. Balancing accuracy, cost, and speed in AI pipelines
4. Handling mixed printed/handwritten content

---

## Future Work

- [ ] Multi-language support (French, Arabic)
- [ ] Drug interaction checking
- [ ] Integration with pharmacy systems
- [ ] Batch processing improvements
- [ ] Mobile camera capture support
- [ ] Historical prescription tracking

---

## Authors

- **Maria Katibi** - katibimaria@gmail.com
- **Branly Djime** - branly.djime@gmail.com

---

## Acknowledgments

- **PaddleOCR** - Open-source OCR engine
- **OpenAI** - GPT-4o and GPT-4o-mini APIs
- **Streamlit** - Web application framework
- Academic supervisors and mentors

---

## License

This project is licensed under the MIT License.

---

**Q.Prescription v1.0** | January 2026
