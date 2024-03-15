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
    doc.close()
    
    return chunks


def dummy_embed_chunk(text:str):
    # This is just a dummy embedding for now
    return np.array([0,0,0,0])


def parse_document(path:str, vector_dir:str):
    chunks = chunk_pdf(path)
    with open(vector_dir+'vector_store.npy', 'wb') as f:
        for c in chunks:
            embed = dummy_embed_chunk(c)
            np.save(f, embed)
    f.close()


def parse_folder(path:str, parsed_dir:str, vector_dir:str):
    if path[-1]!='/': path.append('/')
    if parsed_dir[-1]!='/': parsed_dir.append('/')
    if vector_dir[-1]!='/': vector_dir.append('/')

    for file in os.listdir(path):
        print('Parsing File:',file)
        parse_document(path+file,vector_dir)
    os.replace(path+file,parsed_dir+file)



if __name__=='__main__':
    #parse_folder('./data/01_raw/','./data/02_processed/','./data/03_vectors/')
    parse_folder('./data/00_test/','./data/02_processed/','./data/03_vectors/')