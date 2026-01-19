"""
OCR Text Parser - Structures OCR text into prescription schema without LLM
Uses regex patterns and heuristics to extract structured data from raw OCR text
"""
import re
from typing import Optional, List, Tuple
from loguru import logger

from extraction.schema import (
    ExtractedPrescription,
    PatientInfo,
    DoctorInfo,
    HospitalInfo,
    MedicationItem,
    HandwritingAnalysis,
    PrescriptionType
)


class OCRTextParser:
    """Parse raw OCR text into structured prescription data"""

    # Common Vietnamese prescription patterns
    PATIENT_NAME_PATTERNS = [
        r"(?:Họ\s*(?:và\s*)?tên|Họ tên|Tên\s*BN|Bệnh\s*nhân|Patient)[:\s]*([^\n\d]+)",
        r"(?:HỌ\s*TÊN|HO TEN)[:\s]*([^\n\d]+)",
    ]

    PATIENT_AGE_PATTERNS = [
        r"(?:Tuổi|Năm sinh|Age|NS)[:\s]*(\d+)",
        r"(\d{1,3})\s*(?:tuổi|t\b)",
    ]

    PATIENT_GENDER_PATTERNS = [
        r"(?:Giới|Giới tính|GT|Sex|Gender)[:\s]*(Nam|Nữ|Male|Female)",
        r"\b(Nam|Nữ)\b",
    ]

    PATIENT_ADDRESS_PATTERNS = [
        r"(?:Địa chỉ|Đ/c|Address)[:\s]*([^\n]+)",
    ]

    DOCTOR_NAME_PATTERNS = [
        r"(?:Bác sĩ|BS|ThS\.?BS|TS\.?BS|PGS\.?TS|GS\.?TS|Dr\.?)[:\s]*([^\n]+)",
        r"(?:Người kê đơn|Bác sĩ điều trị)[:\s]*([^\n]+)",
    ]

    HOSPITAL_PATTERNS = [
        r"(?:Bệnh viện|BV|Phòng khám|PK|Hospital|Clinic)[:\s]*([^\n]+)",
        r"(Bệnh Viện[^\n]+)",
    ]

    DEPARTMENT_PATTERNS = [
        r"(?:Khoa|K\.|Department)[:\s]*([^\n]+)",
    ]

    DIAGNOSIS_PATTERNS = [
        r"(?:Chẩn đoán|CĐ|Diagnosis)[:\s]*([^\n]+)",
        r"(?:CHẨN ĐOÁN)[:\s]*([^\n]+)",
    ]

    DATE_PATTERNS = [
        r"(?:Ngày|Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    ]

    PRESCRIPTION_NUMBER_PATTERNS = [
        r"(?:Số|Mã|SĐT|No\.?|Number)[:\s]*(\d{6,})",
        r"(?:Mã đơn thuốc)[:\s]*(\d+)",
    ]

    # Medication patterns - look for numbered items or bullet points
    MEDICATION_PATTERNS = [
        r"(\d+)[.\)]\s*([^\n]+)",  # "1. Medication name..."
        r"[-•]\s*([^\n]+)",  # "- Medication name..."
    ]

    # Common medication keywords to identify medication lines
    MEDICATION_KEYWORDS = [
        "mg", "ml", "viên", "tablet", "capsule", "gói", "ống", "chai",
        "ngày", "lần", "sáng", "trưa", "tối", "sau ăn", "trước ăn",
        "uống", "tiêm", "bôi", "nhỏ"
    ]

    def __init__(self):
        """Initialize parser"""
        pass

    def parse(self, ocr_text: str, ocr_confidence: float) -> ExtractedPrescription:
        """
        Parse OCR text into structured prescription

        Args:
            ocr_text: Raw text from OCR
            ocr_confidence: OCR confidence score

        Returns:
            ExtractedPrescription object
        """
        logger.info("Parsing OCR text into structured prescription...")

        # Extract all components
        patient = self._extract_patient_info(ocr_text)
        doctor = self._extract_doctor_info(ocr_text)
        hospital = self._extract_hospital_info(ocr_text)
        medications = self._extract_medications(ocr_text)
        diagnosis = self._extract_diagnosis(ocr_text)
        issue_date = self._extract_date(ocr_text)
        prescription_number = self._extract_prescription_number(ocr_text)

        # Determine prescription type based on confidence
        if ocr_confidence < 0.5:
            prescription_type = PrescriptionType.HANDWRITTEN
        elif ocr_confidence < 0.7:
            prescription_type = PrescriptionType.MIXED
        else:
            prescription_type = PrescriptionType.PRINTED

        # Create prescription object
        prescription = ExtractedPrescription(
            document_type="prescription",
            prescription_type=prescription_type,
            prescription_number=prescription_number,
            issue_date=issue_date,
            patient=patient,
            doctor=doctor,
            hospital=hospital,
            diagnosis=diagnosis,
            medications=medications,
            extraction_method="ocr_only",
            confidence_score=ocr_confidence,
            ocr_confidence=ocr_confidence,
            llm_enhanced=False,
            total_items=len(medications),
            ocr_text=ocr_text
        )

        logger.info(f"OCR parsing complete: {len(medications)} medications found")
        return prescription

    def _extract_patient_info(self, text: str) -> PatientInfo:
        """Extract patient information from text"""
        name = self._find_first_match(text, self.PATIENT_NAME_PATTERNS)
        age = self._find_first_match(text, self.PATIENT_AGE_PATTERNS)
        gender = self._find_first_match(text, self.PATIENT_GENDER_PATTERNS)
        address = self._find_first_match(text, self.PATIENT_ADDRESS_PATTERNS)

        return PatientInfo(
            name=self._clean_text(name),
            age=age,
            gender=gender,
            address=self._clean_text(address)
        )

    def _extract_doctor_info(self, text: str) -> DoctorInfo:
        """Extract doctor information from text"""
        name = self._find_first_match(text, self.DOCTOR_NAME_PATTERNS)

        # Try to extract title from name
        title = None
        if name:
            title_match = re.match(r"(ThS\.?BS|TS\.?BS|PGS\.?TS|GS\.?TS|BS|Dr\.?)", name)
            if title_match:
                title = title_match.group(1)
                name = name[len(title):].strip()

        return DoctorInfo(
            name=self._clean_text(name),
            title=title
        )

    def _extract_hospital_info(self, text: str) -> HospitalInfo:
        """Extract hospital/clinic information from text"""
        name = self._find_first_match(text, self.HOSPITAL_PATTERNS)
        department = self._find_first_match(text, self.DEPARTMENT_PATTERNS)

        return HospitalInfo(
            name=self._clean_text(name),
            department=self._clean_text(department)
        )

    def _extract_medications(self, text: str) -> List[MedicationItem]:
        """Extract medications from text"""
        medications = []
        lines = text.split('\n')

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check if line looks like a medication
            if self._is_medication_line(line):
                med = self._parse_medication_line(line)
                if med:
                    medications.append(med)

        # If no medications found with patterns, try to find any lines with medication keywords
        if not medications:
            for line in lines:
                line = line.strip()
                if any(kw in line.lower() for kw in self.MEDICATION_KEYWORDS):
                    med = self._parse_medication_line(line)
                    if med:
                        medications.append(med)

        return medications

    def _is_medication_line(self, line: str) -> bool:
        """Check if a line appears to be a medication entry"""
        # Check for numbered format
        if re.match(r"^\d+[.\)]\s*", line):
            return True

        # Check for bullet points
        if line.startswith(('-', '•', '*')):
            return True

        # Check for medication keywords
        line_lower = line.lower()
        keyword_count = sum(1 for kw in self.MEDICATION_KEYWORDS if kw in line_lower)
        if keyword_count >= 2:
            return True

        return False

    def _parse_medication_line(self, line: str) -> Optional[MedicationItem]:
        """Parse a medication line into MedicationItem"""
        # Remove leading number/bullet
        line = re.sub(r"^[\d]+[.\)]\s*", "", line)
        line = re.sub(r"^[-•*]\s*", "", line)
        line = line.strip()

        if not line or len(line) < 3:
            return None

        # Try to extract components
        name = line
        dosage = None
        quantity = None
        frequency = None

        # Extract dosage (e.g., "400mg", "100ml")
        dosage_match = re.search(r"(\d+\s*(?:mg|ml|g|mcg|iu))", line, re.IGNORECASE)
        if dosage_match:
            dosage = dosage_match.group(1)

        # Extract quantity (e.g., "30 viên", "x20")
        qty_match = re.search(r"(?:x|SL:?|số lượng:?)\s*(\d+)|(\d+)\s*(?:viên|tablet|capsule|gói|ống)", line, re.IGNORECASE)
        if qty_match:
            quantity = qty_match.group(1) or qty_match.group(2)

        # Extract frequency (e.g., "ngày 2 lần", "sáng 1 viên")
        freq_patterns = [
            r"(ngày\s*(?:uống\s*)?\d+\s*(?:lần|viên)[^\n]*)",
            r"(sáng\s*\d+\s*viên[^\n]*)",
            r"((?:sáng|trưa|tối)[^\n]*viên[^\n]*)",
            r"(sau ăn[^\n]*)",
            r"(trước ăn[^\n]*)",
        ]
        for pattern in freq_patterns:
            freq_match = re.search(pattern, line, re.IGNORECASE)
            if freq_match:
                frequency = freq_match.group(1).strip()
                break

        # Clean up name - remove extracted parts
        if dosage:
            name = name.replace(dosage, "")
        if frequency:
            name = name.replace(frequency, "")

        # Further clean name
        name = re.sub(r"\s+", " ", name).strip()
        name = re.sub(r"[,\s]+$", "", name)

        if len(name) < 2:
            name = line  # Use original if cleaning removed too much

        return MedicationItem(
            name=name,
            dosage=dosage,
            quantity=quantity,
            frequency=frequency,
            is_handwritten=None  # Unknown from OCR alone
        )

    def _extract_diagnosis(self, text: str) -> Optional[str]:
        """Extract diagnosis from text"""
        diagnosis = self._find_first_match(text, self.DIAGNOSIS_PATTERNS)
        return self._clean_text(diagnosis)

    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text"""
        return self._find_first_match(text, self.DATE_PATTERNS)

    def _extract_prescription_number(self, text: str) -> Optional[str]:
        """Extract prescription number from text"""
        return self._find_first_match(text, self.PRESCRIPTION_NUMBER_PATTERNS)

    def _find_first_match(self, text: str, patterns: List[str]) -> Optional[str]:
        """Find first matching pattern in text"""
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            if match:
                return match.group(1).strip()
        return None

    def _clean_text(self, text: Optional[str]) -> Optional[str]:
        """Clean extracted text"""
        if not text:
            return None
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Remove trailing punctuation
        text = re.sub(r"[:\s,]+$", "", text)
        return text if text else None


# Global parser instance
ocr_parser = OCRTextParser()
