import fitz
import numpy as np
import os


def chunk_pdf(filename:str):
    doc = fitz.open(filename)
    toc = doc.get_toc()

    toc_id = 0
    chunks = []
    running_text = ''
    for page in doc:
        blocks = page.get_text('blocks')
        
        for b in blocks:
            text = b[4].replace('\n',' ').strip()
            if toc_id<len(toc) and text==toc[toc_id][1]:
                # save current running text
                chunks.append(running_text)

                running_text=''
                toc_id+=1

            elif text=='References':
                break
            else:
                running_text+='\n'+text
    chunks.append(running_text)
    
    return chunks


def embed_chunk(text:str):
    raise NotImplementedError


def parse_document(path:str):
    chunks = chunk_pdf(path)
    with open('test.npy', 'wb') as f:
        for c in chunks:
            embed = embed_chunk(c)
            np.save(f, embed)


def parse_folder(path:str):
    if path[-1]!='/': path.append('/')

    for file in os.listdir(path):
        parse_document(path+file)


if __name__=='__main__':
    parse_folder('./data/01_raw/')
    _ = chunk_pdf(filename='./data/01_raw/0shotTTS.pdf')