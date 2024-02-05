# import libraries
import ocrmypdf
import pandas as pd
import fitz #!pip install PyMuPDF\
import os

def extract_pdf():
    RAW_DATA = './data/01_raw/'
    PROCESSED_DATA = './data/02_processed/'
    file_list = [f for f in os.listdir(path=RAW_DATA) if f.endswith('.pdf') or f.endswith('.PDF')]
    processed_list = [f for f in os.listdir(path=PROCESSED_DATA) if f.endswith('.pdf') or f.endswith('.PDF')]

    #file_list = list(set(file_list) - set(processed_list)) # won't work due to OCR prepend tag

    '''
    main ocr code, which create new pdf file with OCR_ ahead its origin filename, 
    and error messege can be find in error_log
    '''
    error_log = {}
    for file in file_list:
        try:
            _ = ocrmypdf.ocr(file, 'OCR_'+file,output_type='pdf',skip_text=True,deskew=True)
        except Exception as e:
            if hasattr(e,'message'):
                error_log[file] = e.message
            else:
                error_log[file] = e
            continue
            
    '''
    extract OCRed PDF using PyMuPDF and save into a pandas dataframe
    '''
    ocr_file_list = [f for f in os.listdir(path=RAW_DATA) if f.startswith('OCR_') ]

    # PDF extraction
    # informations we want to extract
    extraction_pdfs = {}

    for file in ocr_file_list:
        #save the results
        pages_df = pages_df = pd.DataFrame(columns=['text'])
        # file reader
        doc = fitz.open(file)
        for page_num in range(doc.pageCount):
            page = doc.loadPage(page_num)
            pages_df = pages_df.append({'text': page.getText('text')}, ignore_index=True)
            
            
        extraction_pdfs[file] = pages_df  
    
    generate_output(extraction_pdfs)

def generate_output(pdf_dict):
    for df in pdf_dict:
        print(df)
        break