import time
import regex as re
import itertools
import string
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm
import multiprocessing as mp
import pandas as pd
import networkit as nk


class deduplicate_docs():
    def __init__(self, minHash_sim_theresh=0.8):
        self.minHash_sim_theresh = minHash_sim_theresh
        self.lsh = None


    # Function to preprocess text and generate features
    def get_features(self, input_args):
        s, width = input_args
        # Lowercase the string
        s = s.lower()

        # Remove all digits (including Persian and Arabic digits)
        s = re.sub(r"[\d٠١٢٣٤٥٦٧٨٩]+", " ", s)

        # Remove Persian day names
        s = s.replace('جمعه', '').replace('شنبه', '')

        # Remove Persian numbers
        persian_numbers = ["یک", "دو", "سه", "چهار", "پنج", "شش", "هفت", "هشت", "نه", "ده"]
        pattern = re.compile(f"({'|'.join(persian_numbers)})")
        s = pattern.sub("", s)

        # Remove punctuation and other specified characters
        s = s.translate(str.maketrans("", "", string.punctuation))
        s = s.translate(str.maketrans("", "", "/:><؟!.،,?"))

        # Remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
        s = re.sub(r"\s+", " ", s.strip())

        # Generate n-grams of the specified width
        return s #list(map(lambda x: "".join(x), ngrams(s, width)))

    # Function to create a MinHash for a document
    def create_minhash(self, text, num_perm=128):
        minhash = MinHash(num_perm=num_perm)
        for txt in text.split():
            minhash.update(txt.encode('utf8'))
        return minhash

    # Preprocess data in parallel
    def preprocess_data_parallel(self, texts, width=3):
        print("Preprocess Data...")
        start_time = time.time()
        with mp.Pool(processes=mp.cpu_count()) as pool:
            args = ((text, width) for text in texts)
            features = pool.map(self.get_features, args)
        print("--- %s seconds ---" % (time.time() - start_time))
        return features

    # Create MinHash objects in parallel
    def create_minhashes_parallel(self, text_list, num_perm=128):
        start_time = time.time()
        print("Creating MinHash...")
        with mp.Pool(processes=mp.cpu_count()) as pool:
            minhashes = pool.starmap(self.create_minhash, [(text, num_perm) for text in text_list])
        print("--- %s seconds ---" % (time.time() - start_time))
        return minhashes


    def query_minhash(self, args):
        doc_id, minhash = args
        result = self.lsh.query(minhash)
        return [(doc_id, other_doc_id) for other_doc_id in result if doc_id != other_doc_id]

    # Main deduplication function
    def hash_deduplication(self, docs):
        # Preprocess texts in parallel
        docs['processed'] = self.preprocess_data_parallel(docs['text'].tolist())

        # Create MinHash objects for each document in parallel
        minhashes_list = self.create_minhashes_parallel(docs['processed'].tolist())
        minhashes = dict(zip(docs['id'], minhashes_list))

        # Create an LSH index
        self.lsh = MinHashLSH(threshold=self.minHash_sim_theresh, num_perm=128)
        for doc_id, minhash in tqdm(minhashes.items()):
            self.lsh.insert(doc_id, minhash)

        # Find similar documents in parallel
        print("Querying LSH...")

        with mp.Pool(processes=mp.cpu_count()) as pool:
            similar_pairs = pool.map(self.query_minhash, minhashes.items())

        # Flatten the list of similar pairs
        similar_docs = [pair for sublist in similar_pairs for pair in sublist]

        # Remove duplicates (since each pair is found twice)
        similar_docs = list(set(frozenset(pair) for pair in similar_docs))

        return similar_docs

    # Function to construct a graph from the set of duplicate pairs
    def construct_graph(self, set_of_duplicate_pairs):
        graph = nk.Graph()
        mapper = {}
        for pair in set_of_duplicate_pairs:
            node1_name, node2_name = pair
            if node1_name not in mapper:
                mapper[node1_name] = graph.addNode()
            if node2_name not in mapper:
                mapper[node2_name] = graph.addNode()
            graph.addEdge(mapper[node1_name], mapper[node2_name])
        return graph, mapper

    # Function to find connected components in the graph
    def find_connected_components(self, graph):
        cc = nk.components.ConnectedComponents(graph)
        cc.run()
        return cc.getComponents(), cc.numberOfComponents()

    # Example usage
    def deduplicate(self, df):

        # Perform deduplication
        similar_hashed = self.hash_deduplication(df)

        # Generate a graph using IDs as nodes and pairs of IDs as edges
        nk.setNumberOfThreads(mp.cpu_count())
        graph, mapper = self.construct_graph(similar_hashed)
        components, n_components = self.find_connected_components(graph)
        print(f"Number of connected components: {n_components}")

        reversed_mapper = {value: key for key, value in mapper.items()}
        duplicates = []
        for compo in components:
            duplicates.append([reversed_mapper[x] for x in compo][1:])

        duplicates = [item for sublist in duplicates for item in sublist]
        # print(f"Duplicate documents: {duplicates}")

        return df[~df['id'].isin(duplicates)].reset_index().drop('processed', axis=1)