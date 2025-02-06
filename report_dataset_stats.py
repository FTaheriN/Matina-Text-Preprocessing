import pandas as pd


def report_stats(docs, doc_source):
    lengths = pd.Series([len(doc.split()) for doc in docs.text.values])
    mean = lengths.mean() #statistics.mean(lengths)
    min_ = lengths.min() #np.min(lengths)
    max_ = lengths.max() #np.max(lengths)
    median = lengths.median() #statistics.median(lengths)
    print("Statistics (word-level) for docs from : "+ '\033[1m' +  doc_source + '\033[0m')
    print("Total of documents: "+'\033[1m' +  str(len(docs)) + '\033[0m')
    print('\033[1m' + "Min " + '\033[0m' + "document length: "+ '\033[1m' + str(min_) + '\033[0m')
    print('\033[1m' + "Max " + '\033[0m' + "document length: "+'\033[1m' +  str(max_) + '\033[0m')
    print('\033[1m' + "Median " + '\033[0m' + "document length: "+'\033[1m' +  str(int(median)) + '\033[0m')
    print('\033[1m' + "Average " + '\033[0m' + "document length: "+'\033[1m' +  str(int(mean)) + '\033[0m')
    print("Total number of words: "+'\033[1m' +  str(lengths.sum()) + '\033[0m')
    print("#"*100)
    return