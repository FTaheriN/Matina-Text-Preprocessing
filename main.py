import os
import glob
import time
import json
import yaml
import random
import argparse
import pandas as pd 

from report_dataset_stats import report_stats

from matinaDeduplicator import deduplicate_docs
from matinaProcessingPipeline import process_pipeline


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def process_documents(config):
    """Process documents using the pipeline."""
    print(f"\nProcessing dataset: {config['pipeline_type'].upper()}")

    files_to_process = []

    # Determine whether to process a single file or multiple files
    if "docs_path" in config and config["docs_path"]:
        files_to_process = [config["docs_path"]]
    elif "all_files_pattern" in config and config["all_files_pattern"]:
        all_files = glob.glob(config["all_files_pattern"])
        if "sample_size" in config and config["sample_size"] > 0:
            random.seed(config["random_seed"])
            files_to_process = random.sample(all_files, min(config["sample_size"], len(all_files)))
        else:
            files_to_process = all_files

    if not files_to_process:
        print("No files to process. Check your YAML configuration.")
        return

    for file_ in files_to_process:
        print(f"Processing file: {file_}")
        docs = pd.read_json(file_, lines=True, compression=config.get("compression", None))
        report_stats(docs, config["pipeline_type"])

        index = file_.split('/')[-1].split('.')[0].split(config["pipeline_type"])[-1]
        pipeline = process_pipeline(config["root_dir"], config["processed_root_dir"], config["dedup_root_dir"], config["pipeline_type"])

        # Run pipeline processing
        raw_books, processed_books = pipeline.parallel_process_docs(
            docs=docs,
            chunk=index,
            num_processes=config["num_processes"],
            short_doc_thresh=config["short_doc_thresh"],
            english_allowed=config["english_allowed"],
            arabic_allowed=config["arabic_allowed"],
            doc_word_length=config["doc_word_length"],
            doc_lang_thresh=config["doc_lang_thresh"],
            doc_num_thresh=config["doc_num_thresh"],
            doc_symb_thresh=config["doc_symb_thresh"],
            doc_stopword_thresh=config["doc_stopword_thresh"],
            shortLine_proportion_thresh=config["shortLine_proportion_thresh"],
            cons_new_lines=config["cons_new_lines"],
            cons_chars=config["cons_chars"],
            non_sense_patterns=config["non_sense_patterns"],
            english_lines=config["english_lines"],
            numeric_lines=config["numeric_lines"],
            symbolic_lines=config["symbolic_lines"],
            personal_info=config["personal_info"],
            emojis=config["emojis"],
            norm_dates=config["norm_dates"],
            norm_numbers=config["norm_numbers"],
            norm_symbols=config["norm_symbols"]
        )

def deduplicate_documents(config):
    """Perform document deduplication."""
    print(f"Deduplicating docs from {config['pipeline_type']}")
    
    processed_docs = pd.read_json(config["processed_docs_path"], lines=True)
    print(processed_docs.columns)
    print("BEFORE DEDUPLICATION")
    report_stats(processed_docs, config["pipeline_type"])

    deduplicator = deduplicate_docs(config["deduplication_threshold"])
    dedup_docs = deduplicator.deduplicate(processed_docs)

    print("AFTER DEDUPLICATION")
    report_stats(dedup_docs, config["pipeline_type"])

    pipeline = process_pipeline(config["root_dir"], config["processed_root_dir"], config["dedup_root_dir"], config["pipeline_type"])
    pipeline.save_deduplicated_docs_jsonl(dedup_docs, chunk=config["output_chunk_suffix"])

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run the document processing or deduplication pipeline with YAML configuration.")
    parser.add_argument("config_paths", type=str, nargs="+", help="Path(s) to the YAML configuration file(s).")
    args = parser.parse_args()

    # Process each provided YAML file
    for config_path in args.config_paths:
        config = load_config(config_path)
        
        if config["mode"] == "process":
            process_documents(config)
        elif config["mode"] == "dedup":
            deduplicate_documents(config)
        else:
            print(f"Invalid mode '{config['mode']}' in {config_path}. Use 'process' or 'dedup'.")

if __name__ == "__main__":
    main()
