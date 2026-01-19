# Q.Prescription - Quick Start Guide

Get started with Q.Prescription in 5 minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

**Note**: PaddleOCR may take a few minutes to install.

## Step 2: Configure API Key

Create a `.env` file:

```bash
cp .env.template .env
```

Edit `.env` and add your OpenAI API key:

```env
OPENAI_API_KEY=sk-your-key-here
```

## Step 3: Run the Application

```bash
streamlit run app.py
```

The app opens at: `http://localhost:8501`

## Step 4: Process Your First Prescription

1. **Upload**: Drag & drop or click to upload prescription images
2. **Options**: Check "Force LLM Vision" for handwritten prescriptions (optional)
3. **Analyze**: Click "Analyze All Prescriptions"
4. **View**: Results appear in the "Library" tab

## What Gets Extracted

Q.Prescription extracts:

- **Patient Info**: Name, age, gender, address, phone
- **Doctor Info**: Name, title, specialty, license number
- **Hospital/Clinic**: Name, address, contact details
- **Medications**: Name, dosage, quantity, frequency, instructions
- **Signature**: Detection, legibility, signer info

## How It Works

The system uses a **3-stage adaptive pipeline**:

| Stage | What Happens |
|-------|--------------|
| **1. OCR** | PaddleOCR extracts text with confidence scoring |
| **2. LLM Processing** | High confidence → LLM Text (cheap), Low confidence → LLM Vision (for handwriting) |
| **3. Signature** | Always uses LLM Vision for signature detection |

## Processing Options

| Option | When to Use |
|--------|-------------|
| **Default** | For printed prescriptions - faster & cheaper |
| **Force LLM Vision** | For handwritten prescriptions - more accurate |

## Tips for Best Results

### For Better OCR
- Use clear, high-resolution images (300+ DPI)
- Ensure good lighting and contrast
- Avoid skewed or rotated images

### For Handwritten Prescriptions
- Enable "Force LLM Vision" option
- Ensure handwriting is reasonably legible
- Good lighting helps significantly

## Common Issues

### "OPENAI_API_KEY not set"
Add your API key to the `.env` file

### Low OCR Confidence
- Image may be blurry or low quality
- System will automatically use LLM Vision for better results

### No Medications Extracted
- Check if handwriting is legible
- Try with "Force LLM Vision" enabled

## View Results

In the **Library** tab, click "View Details" to see:

- **Preview**: Original prescription image
- **Patient & Doctor**: Extracted information
- **Medications**: List with dosage, quantity, frequency
- **Signature**: Detection results with confidence
- **Processing Info**: Which method was used and why

## Export Data

Click "Export All as JSON" in the Library tab to download structured data.

## Test Dataset

The project includes sample prescriptions in `prescription_dataset/`:

| Folder | Type | Count |
|--------|------|-------|
| `printed/` | Fully printed prescriptions | 99 |
| `mixed/` | Printed + handwritten | 17 |
| `handwriting/` | Fully handwritten | 5 |

---

**Happy prescription processing!**
