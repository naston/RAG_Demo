import numpy as np

def parse_files(EM, doc_map):
    pass


def create_index(EM, doc_map, exact=False):
    pass


if __name__=="__main__":
    with open() as f:
        doc_map = f.load()
    f.close()

    EM = None
    doc_map = parse_files(EM, doc_map)

    index = create_index(EM, doc_map)

    # save index to 06