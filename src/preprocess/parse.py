import os
import json
import uuid
import fitz
import numpy as np


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
    chunk_ids = []
    with open(vector_dir+'vector_store.npy', 'wb') as f:
        for c in chunks:
            embed = dummy_embed_chunk(c)
            np.save(f, embed)

            chunk_ids.append(uuid.uuid4().hex)
    f.close()
    return chunk_ids


def parse_folder(path:str, parsed_dir:str, vector_dir:str, doc_map:dict):
    if path[-1]!='/': path.append('/')
    if parsed_dir[-1]!='/': parsed_dir.append('/')
    if vector_dir[-1]!='/': vector_dir.append('/')

    for file in os.listdir(path):
        print('Parsing File:',file)
        doc_uuid = uuid.uuid4().hex
        new_file = doc_uuid + '.' + file.split('.')[-1]

        chunk_ids = parse_document(path+file,vector_dir)

        embed_index = len(doc_map)
        for i, chunk_id in enumerate(chunk_ids):
            doc_map[embed_index+i]={
                            'chunk':chunk_id,
                            'doc':doc_uuid}

        os.replace(path+file,parsed_dir+new_file)


if __name__=='__main__':
    # './data/01_raw/'
    with open('./data/doc_map.json','r+') as f:
        doc_map = json.load(f)
        doc_map = parse_folder('./data/00_test/','./data/02_processed/','./data/03_vectors/', doc_map)

        f.seek(0)
        json.dump(doc_map, f)
    f.close()