from typing import List, Dict, Union
import os
import PyPDF2


class PDFProcessor:
    
    def __init__(self) -> None:
        pass

    @staticmethod
    def process_pdf_file(
        file_info: Dict, 
        document_text: List,
        pdfs_path: str = 'data/') -> List:
        """
        Process content of a PDF file and append information to the document_text list.

        Parameters:
        - file_info (Dict): Information about the PDF file.
        - document_text (List): List containing document information.
        - pdfs_path (str): Path to the folder containing PDF files (default is 'data/').

        Returns:
        - List: Updated document_text list.
        """
        file_title = file_info["title"]
        
        if file_title.split('.')[-1] == 'pdf':
            print(f'⛳️ File Name: {file_title}')
            
            pdf_path = os.path.join(pdfs_path, file_title)
            pdf_reader = PyPDF2.PdfReader(pdf_path)
            pages_amount = len(pdf_reader.pages)
            print(f'Amount of pages: {pages_amount}')
            
            for i, page in enumerate(pdf_reader.pages):
                document_text.append([file_title, file_info['embedLink'], i + 1, page.extract_text()])
        return document_text