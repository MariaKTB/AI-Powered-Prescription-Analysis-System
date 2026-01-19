"""
DocuVault - Document Schema
Comprehensive schema for various document types including medical prescriptions
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import date
from enum import Enum

# ============================================
# MEDICAL PRESCRIPTION SCHEMA
# ============================================

class PrescriptionType(str, Enum):
    """Type of prescription document"""
    HANDWRITTEN = "handwritten"
    PRINTED = "printed"
    MIXED = "mixed"
    DIGITAL = "digital"


class SignatureInfo(BaseModel):
    """Information about a signature on the document"""
    is_present: bool = Field(default=False, description="Whether a signature is detected")
    signer_name: Optional[str] = Field(default=None, description="Name of the signer if legible")
    signer_title: Optional[str] = Field(default=None, description="Title (e.g., 'Doctor', 'Physician')")
    location: Optional[str] = Field(default=None, description="Where on document (e.g., 'bottom right')")
    is_legible: Optional[bool] = Field(default=None, description="Whether signature is legible")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence in signature detection")


class MedicationItem(BaseModel):
    """A single medication prescribed"""
    name: str = Field(description="Name of the medication")
    dosage: Optional[str] = Field(default=None, description="Dosage amount (e.g., '400mg', '100mg')")
    quantity: Optional[str] = Field(default=None, description="Quantity prescribed (e.g., '30 tablets')")
    frequency: Optional[str] = Field(default=None, description="How often to take (e.g., '1 viên sau ăn', 'twice daily')")
    duration: Optional[str] = Field(default=None, description="Treatment duration (e.g., '7 days', '1 month')")
    instructions: Optional[str] = Field(default=None, description="Special instructions")
    is_handwritten: Optional[bool] = Field(default=None, description="Whether this item was handwritten")


class PatientInfo(BaseModel):
    """Patient information from prescription"""
    name: Optional[str] = Field(default=None, description="Patient's full name")
    age: Optional[str] = Field(default=None, description="Patient's age")
    gender: Optional[str] = Field(default=None, description="Patient's gender (Nam/Nữ or Male/Female)")
    address: Optional[str] = Field(default=None, description="Patient's address")
    phone: Optional[str] = Field(default=None, description="Patient's phone number")
    patient_id: Optional[str] = Field(default=None, description="Patient ID or medical record number")


class DoctorInfo(BaseModel):
    """Doctor/Prescriber information"""
    name: Optional[str] = Field(default=None, description="Doctor's name")
    title: Optional[str] = Field(default=None, description="Doctor's title (e.g., 'ThS.BS', 'Dr.')")
    specialty: Optional[str] = Field(default=None, description="Medical specialty")
    license_number: Optional[str] = Field(default=None, description="Medical license number")
    phone: Optional[str] = Field(default=None, description="Doctor's contact phone")
    signature: Optional[SignatureInfo] = Field(default=None, description="Doctor's signature info")


class HospitalInfo(BaseModel):
    """Hospital/Clinic information"""
    name: Optional[str] = Field(default=None, description="Hospital or clinic name")
    department: Optional[str] = Field(default=None, description="Department (Khoa)")
    address: Optional[str] = Field(default=None, description="Hospital address")
    phone: Optional[str] = Field(default=None, description="Hospital phone")
    pharmacy_counter: Optional[str] = Field(default=None, description="Pharmacy counter number")


class HandwritingAnalysis(BaseModel):
    """Analysis of handwritten content"""
    has_handwritten_content: bool = Field(default=False, description="Whether document contains handwriting")
    handwritten_sections: List[str] = Field(default_factory=list, description="List of handwritten sections identified")
    ocr_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="OCR confidence for handwritten parts")
    llm_interpretation: Optional[str] = Field(default=None, description="LLM's interpretation of hard-to-read text")
    unclear_text: List[str] = Field(default_factory=list, description="Text that couldn't be clearly read")


class ExtractedPrescription(BaseModel):
    """Main medical prescription structure"""
    document_type: str = Field(
        default="prescription",
        description="Type of document - always 'prescription' for this schema"
    )
    prescription_type: Optional[PrescriptionType] = Field(
        default=None,
        description="Whether prescription is handwritten, printed, or mixed"
    )

    # Identifiers
    prescription_number: Optional[str] = Field(default=None, description="Prescription ID/number")
    barcode: Optional[str] = Field(default=None, description="Barcode number if present")

    # Date information
    issue_date: Optional[str] = Field(default=None, description="Date prescription was issued")

    # People involved
    patient: PatientInfo = Field(default_factory=PatientInfo)
    doctor: DoctorInfo = Field(default_factory=DoctorInfo)
    hospital: HospitalInfo = Field(default_factory=HospitalInfo)

    # Medical content
    diagnosis: Optional[str] = Field(default=None, description="Diagnosis (Chẩn đoán)")
    medications: List[MedicationItem] = Field(default_factory=list, description="List of prescribed medications")

    # Signatures
    doctor_signature: Optional[SignatureInfo] = Field(default=None, description="Doctor's signature")

    # Handwriting analysis
    handwriting_analysis: Optional[HandwritingAnalysis] = Field(
        default=None,
        description="Analysis of handwritten content"
    )

    # Additional info
    notes: Optional[str] = Field(default=None, description="Additional notes or instructions")
    total_items: Optional[int] = Field(default=None, description="Total number of medication items")

    # Extraction metadata
    extraction_method: Optional[str] = Field(
        default=None,
        description="How data was extracted: 'ocr_only', 'llm_vision', 'ocr_plus_llm'"
    )
    confidence_score: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Overall extraction confidence"
    )
    ocr_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="OCR-specific confidence"
    )
    llm_enhanced: bool = Field(
        default=False,
        description="Whether LLM vision was used to enhance extraction"
    )
    ocr_text: Optional[str] = Field(default=None, description="Raw OCR text extracted from the document")

    class Config:
        json_schema_extra = {
            "example": {
                "document_type": "prescription",
                "prescription_type": "mixed",
                "prescription_number": "22001918442",
                "issue_date": "2022-09-14",
                "patient": {
                    "name": "Nguyễn Việt Ghi",
                    "age": "65",
                    "gender": "Nam",
                    "address": "Xã Bình Thạnh Huyện Bình Sơn Quảng Ngãi"
                },
                "doctor": {
                    "name": "Bùi Văn Đoàn",
                    "title": "ThS.BS"
                },
                "hospital": {
                    "name": "Bệnh Viện TW Huế",
                    "department": "K. Khám bệnh"
                },
                "diagnosis": "Viêm gan C mạn",
                "medications": [
                    {
                        "name": "Epclusa (400MG + 100MG)(Sofosbuvir 400mg+ Velpatasvir 100mg)",
                        "quantity": "30",
                        "frequency": "Ngày uống sáng 1 viên sau ăn"
                    },
                    {
                        "name": "ANBALIV 400mg (Silymarin)",
                        "quantity": "60",
                        "frequency": "Ngày uống sáng 1 viên, tối 1 viên sau ăn"
                    }
                ],
                "doctor_signature": {
                    "is_present": True,
                    "signer_name": "Bùi Văn Đoàn",
                    "location": "bottom right",
                    "is_legible": True
                },
                "extraction_method": "ocr_plus_llm",
                "confidence_score": 0.85,
                "llm_enhanced": True
            }
        }


