import numpy as np
import faiss


class VectorIndex(object):
    def __init__(self) -> None:
        super().__init__()
        self.index = None


    def create_index(self, vector_dim:int, m:int=8, nlist:int=256, nbits:int=8, nprobe:int=64):
        assert vector_dim % m == 0

        coarseQuantizer = faiss.IndexFlatL2(vector_dim)
        self.index = faiss.IndexIVFPQ(coarseQuantizer, vector_dim, nlist, m, nbits)
        self.index.nprobe = nprobe


    def is_trained(self):
        return self.index.is_trained()


    def train_index(self, vector_store:np.ndarray):
        assert self.index is not None
        self.index.train(vector_store)


    def save_index(self, path:str):
        faiss.write_index(self.index, path)


    def load_index(self,path:str):
        self.index = faiss.read_index(path)


    def add_vectors(self, vector_store:np.ndarray):
        self.index.add(vector_store)


    def search_vectors(self, query_vect:np.ndarray, k=10):
        #res = faiss.StandardGpuResources()
        #gpu_index = faiss.index_cpu_to_gpu(res, 0, self.index)
        #gpu_index.add(vector_store)

        distance, context_vects = self.index.search(query_vect, k)

        return distance, context_vects


if __name__=='__main__':
    vector_store = np.random.rand(10_000, 768)

    index = VectorIndex()
    index.create_index(vector_store.shape[1])

    index.train_index(vector_store)
    index.add_vectors(vector_store)

    query = np.random.rand(1,768)

    D, I = index.search_vectors(query)
    print(D[0,0])
    print(I)