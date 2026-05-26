"""PDF Content Extraction Tool for CrewAI.

This tool provides a CrewAI-compatible wrapper for extracting text from PDF files.
Note: The primary PDF processing is handled by the RAG pipeline (rag.py).
This tool is available as an optional agent capability.
"""

from crewai.tools import BaseTool
from pydantic import Field

import fitz  # PyMuPDF


class ExtractPDFContentTool(BaseTool):
    name: str = "Extract PDF Content"
    description: str = (
        "Extracts and returns the full text content from a PDF file. "
        "Provide the absolute file path to the PDF."
    )
    pdf_path: str = Field(default="", description="Path to the PDF file")

    def _run(self, pdf_path: str = "") -> str:
        """Extract text from a PDF file using PyMuPDF.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            The extracted text content, or an error message.
        """
        path = pdf_path or self.pdf_path
        if not path:
            return "Error: No PDF path provided."

        try:
            doc = fitz.open(path)
            full_text = ""
            for page_num, page in enumerate(doc):
                page_text = page.get_text()
                if page_text.strip():
                    full_text += f"\n--- Page {page_num + 1} ---\n"
                    full_text += page_text
            doc.close()

            if not full_text.strip():
                return "Warning: PDF appears to contain no extractable text (may be scanned/image-based)."

            return full_text.strip()
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
