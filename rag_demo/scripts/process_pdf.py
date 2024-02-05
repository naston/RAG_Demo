# import libraries
import ocrmypdf
import pandas as pd
import fitz #!pip install PyMuPDF\
import os

from pathlib import Path
from llama_index import download_loader

def pdf_direct():
    PROCESSED_DATA = './data/02_processed/'
    PDFReader = download_loader("PDFReader")

    loader = PDFReader()
    documents = loader.load_data(file=Path(PROCESSED_DATA))

def extract_pdf():
    RAW_DATA = './data/01_raw/'
    PROCESSED_DATA = './data/02_processed/'
    file_list = [f for f in os.listdir(path=RAW_DATA) if f.endswith('.pdf') or f.endswith('.PDF')]
    processed_list = [f[4:] for f in os.listdir(path=PROCESSED_DATA) if f.endswith('.pdf') or f.endswith('.PDF')]
    
    print(len(file_list),len(processed_list))

    file_list = list(set(file_list) - set(processed_list)) # won't work due to OCR prepend tag
    print(file_list)
    '''
    main ocr code, which create new pdf file with OCR_ ahead its origin filename, 
    and error messege can be find in error_log
    '''
    error_log = {}
    for file in file_list:
        break
        print('OCR on file',file)
        if file == file_list[3]:
            continue
        try:
            _ = ocrmypdf.ocr(RAW_DATA+file, PROCESSED_DATA+'OCR_'+file,output_type='pdf',skip_text=True,deskew=True)
        except Exception as e:
            if hasattr(e,'message'):
                error_log[file] = e.message
            else:
                error_log[file] = e
            continue

    print(error_log)
            
    '''
    extract OCRed PDF using PyMuPDF and save into a pandas dataframe
    '''
    ocr_file_list = [f for f in os.listdir(path=PROCESSED_DATA) if f.startswith('OCR_') ]
    #ocr_file_list = ['OCR_'+f for f in file_list]

    # PDF extraction
    # informations we want to extract
    text = []

    for file in ocr_file_list:
        # file reader
        doc = fitz.open(PROCESSED_DATA+file)
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            text.append(page.get_text('text'))
    print(len(text)) 
    
    return text

def generate_output(pdf_dict):
    print('gen output...')
    for df in pdf_dict.values():
        print(df)
        break
