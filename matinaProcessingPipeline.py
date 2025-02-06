import os
import json
import time
import progressbar
import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
from chardet import detect
from report_dataset_stats import report_stats
from matinaPreprocessor import preprocess_docs
from matinaDeduplicator import deduplicate_docs


# get file encoding type
def get_encoding_type(file):
    with open(file, 'rb') as f:
        rawdata = f.read()
    return detect(rawdata)['encoding']


class process_pipeline():
    def __init__(self, root_dir, processed_root_dir, dedup_root_dir, source_dir):
        self.root_dir = root_dir
        self.processed_root_dir = processed_root_dir
        self.dedup_root_dir = dedup_root_dir
        self.source_dir = source_dir
        self.doc_processor = preprocess_docs(self.source_dir)
    
    
    def load_files_text(self):
        docs = []
        names = []
        pathlist = Path(os.path.join(self.root_dir, self.source_dir)).rglob("*.txt")
        for path in pathlist:
            names.append(str(path).split('.txt')[0].split('/')[-1])
            # with open(path, 'r') as f:
            byte = b'\xff'
            with open(path,'rb') as f:
                file_content = f.read()
            if byte in file_content:
                pass
            else:
                with open(path, 'r', encoding = get_encoding_type(path), errors='ignore') as f:
                    docs.append(f.read())
        print(len(docs))
        ids = [self.source_dir+str(i) for i in range(len(docs))]
        return pd.DataFrame(zip(ids, names, docs), columns=['id','file_name','text'])
    
    
    def load_files_jsonl(self, file_path):
        return pd.read_json(file_path, lines=True)
        
        
    def save_processed_docs_jsonl(self, df, chunk):
        try:
            df = df.drop('processed', axis=1)
        except:
            pass
        os.makedirs(self.processed_root_dir, exist_ok=True)
        df.to_json(os.path.join(self.processed_root_dir, self.source_dir+str(chunk)+'.jsonl.gz'), orient='records', lines=True, force_ascii=False, compression='gzip')
        # df.to_json(os.path.join(self.dedup_root_dir, self.source_dir+str(chunk)+'.jsonl'), orient='records', lines=True, force_ascii=False)
        print(f"Processed file successfully saved to {os.path.join(self.processed_root_dir, self.source_dir+str(chunk)+'.jsonl')}")
        
        
    def save_deduplicated_docs_jsonl(self, df, chunk=''):
        os.makedirs(self.dedup_root_dir, exist_ok=True)
        df.to_json(os.path.join(self.dedup_root_dir, self.source_dir+str(chunk)+'.jsonl'), orient='records', lines=True, force_ascii=False)
        print(f"Processed file successfully saved to {os.path.join(self.dedup_root_dir, self.source_dir+str(chunk)+'.jsonl')}")
        

    def process_chunk(self, chunk):
        processed_docs = []
        processed_titles = []
        processed_ids = []
        for i, row in chunk.iterrows():
            # doc = self.doc_processor.eliminate_document_level(row['text'])
            # if doc:
            #     processed_doc = self.doc_processor.preprocess_character_level(doc)
            #     if self.doc_processor.eliminate_if_short(processed_doc):
            processed_doc = self.doc_processor.preprocess_character_level(row['text'])
            if self.doc_processor.eliminate_if_short(processed_doc):
                doc = self.doc_processor.eliminate_document_level(processed_doc)
                if doc:
                    processed_docs.append([(row['id'], doc)])
        return processed_docs

    def parallel_process_docs(self, docs, chunk, num_processes=mp.cpu_count(),
                    english_allowed=False, arabic_allowed=False,
                    doc_lang_thresh=0.5, doc_num_thresh=0.8, doc_symb_thresh=0.8, 
                    doc_word_length=[3,10], doc_stopword_thresh=0.1, short_doc_thresh=150,
                    shortLine_proportion_thresh=0.8, english_lines=True,
                    cons_new_lines=True, cons_chars=True, non_sense_patterns=True, 
                    numeric_lines=True, symbolic_lines=True, personal_info=True, emojis=False,
                    cons_newline_thresh=2, cons_chars_thresh=3, numeric_lines_thresh=0.8,
                    symbolic_lines_thresh=0.8, repeated_lines_thresh=10,
                    norm_dates=True, norm_numbers=True, norm_symbols=True):
    
        print("Processing docs from ", self.source_dir)
        self.doc_processor = preprocess_docs(self.source_dir,
                        english_allowed, arabic_allowed,
                        doc_lang_thresh, doc_num_thresh, doc_symb_thresh, 
                        doc_word_length, doc_stopword_thresh, short_doc_thresh,
                        shortLine_proportion_thresh, english_lines,
                        cons_new_lines, cons_chars, non_sense_patterns, 
                        numeric_lines, symbolic_lines, personal_info, emojis,
                        cons_newline_thresh, cons_chars_thresh, numeric_lines_thresh,
                        symbolic_lines_thresh, repeated_lines_thresh,
                        norm_dates, norm_numbers, norm_symbols)
        
        
        # print("Loading file...")
        # docs = self.load_files_text().reset_index()
        # report_stats(docs, self.source_dir)
        
        # Split the dataframe into chunks
        chunk_size = len(docs) // num_processes + 1
        chunks = [docs.iloc[i:i + chunk_size] for i in range(0, len(docs), chunk_size)]

        with mp.Pool(processes=num_processes) as pool:
            # Initialize the progress bar
            widgets = [
                'Processing: ', progressbar.Percentage(),
                ' ', progressbar.Bar(marker='*', left='[', right=']'),
                ' ', progressbar.Counter(), '/'+str(len(docs)),
                ' ', progressbar.Timer(),
                ' ', progressbar.ETA()
            ]
            bar = progressbar.ProgressBar(widgets=widgets, maxval=len(docs))
            bar.start()

            # Process chunks in parallel
            results = []
            for result in pool.imap(self.process_chunk, chunks):
                results.extend(result)
                bar.update(len(results))

        # Combine the results into a dataframe
        processed_docs = [item for sublist in results for item in sublist]
        cleaned_docs = pd.DataFrame(processed_docs, columns=['id', 'text'])
        # report_stats(cleaned_docs, self.source_dir)
        self.save_processed_docs_jsonl(cleaned_docs, chunk)       
        return docs, cleaned_docs
        
    def process_docs(self,
                    english_allowed=False, arabic_allowed=False,
                    doc_lang_thresh=0.5, doc_num_thresh=0.8, doc_symb_thresh=0.8, 
                    doc_word_length=[3,10], doc_stopword_thresh=0.1, short_doc_thresh=150,
                    shortLine_proportion_thresh=0.8, english_lines=True,
                    cons_new_lines=True, cons_chars=True, non_sense_patterns=True, 
                    numeric_lines=True, symbolic_lines=True, personal_info=True, emojis=False,
                    cons_newline_thresh=2, cons_chars_thresh=3, numeric_lines_thresh=0.8,
                    symbolic_lines_thresh=0.8, repeated_lines_thresh=10,
                    norm_dates=True, norm_numbers=True, norm_symbols=True):
        
        print("Processing docs from ", self.source_dir)
        self.doc_processor = preprocess_docs(self.source_dir,
                    english_allowed, arabic_allowed,
                    doc_lang_thresh, doc_num_thresh, doc_symb_thresh, 
                    doc_word_length, doc_stopword_thresh, short_doc_thresh,
                    shortLine_proportion_thresh, english_lines,
                    cons_new_lines, cons_chars, non_sense_patterns, 
                    numeric_lines, symbolic_lines, personal_info, emojis,
                    cons_newline_thresh, cons_chars_thresh, numeric_lines_thresh,
                    symbolic_lines_thresh, repeated_lines_thresh, 
                    norm_dates, norm_numbers, norm_symbols)
        
        print("Loading file...")
        docs = self.load_files_text().reset_index()
        report_stats(docs, self.source_dir)
        
        processed_docs, processed_titles, processed_ids = [], [], []
        print("Processing and cleaning...")
        # bar = progressbar.ProgressBar(maxval=len(docs))
        widgets = [
            'Processing: ', progressbar.Percentage(),
            ' ', progressbar.Bar(marker='*', left='[', right=']'),
            ' ', progressbar.Counter(), '/'+str(len(docs)),
            ' ', progressbar.Timer(),
            ' ', progressbar.ETA()
        ]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(docs))
        bar.start()
        for i, row in docs.iterrows():
            doc = self.doc_processor.eliminate_document_level(row['text'])
            if doc:
                processed_doc = self.doc_processor.preprocess_character_level(doc)
                if self.doc_processor.eliminate_if_short(processed_doc):
                    processed_docs.append(processed_doc)
                    processed_titles.append(row['title'])
                    processed_ids.append(row['id'])
            bar.update(i)
        cleaned_df = pd.DataFrame(zip(processed_ids, processed_titles, processed_docs), columns=['id','title', 'text'])
        self.save_processed_docs_jsonl(cleaned_df)       
        return docs, cleaned_df