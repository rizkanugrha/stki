import numpy as np

def calculate_precision_recall_f1(retrieved_docs, relevant_docs):
    """Menghitung Precision, Recall, dan F1-Score."""
    # 
    retrieved_set = set(retrieved_docs)
    relevant_set = set(relevant_docs)
    
    true_positives = len(retrieved_set.intersection(relevant_set))
    
    precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0.0
    recall = true_positives / len(relevant_set) if len(relevant_set) > 0 else 0.0
    
    f1 = 0.0
    if (precision + recall) > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
        
    return precision, recall, f1

def precision_at_k(retrieved_docs, relevant_docs, k):
    """Menghitung Precision@k."""
    # 
    if not retrieved_docs:
        return 0.0
    
    retrieved_at_k = retrieved_docs[:k]
    relevant_set = set(relevant_docs)
    
    true_positives_at_k = len(set(retrieved_at_k).intersection(relevant_set))
    
    return true_positives_at_k / k

def average_precision(retrieved_docs, relevant_docs):
    """Menghitung Average Precision (AP)."""
    # (bagian dari MAP@k)
    relevant_set = set(relevant_docs)
    if not relevant_set:
        return 0.0
        
    ap = 0.0
    true_positives = 0
    
    for i, doc_id in enumerate(retrieved_docs):
        if doc_id in relevant_set:
            true_positives += 1
            ap += (true_positives / (i + 1)) # Standar AP

    return ap / len(relevant_set) if relevant_set else 0.0

def mean_average_precision(query_results, gold_standard, k=10):
    """Menghitung Mean Average Precision (MAP@k)."""
    # 
    aps = []
    for query_id, relevant_docs in gold_standard.items():
        if query_id in query_results:
            retrieved_docs = query_results[query_id]
            # Ambil hanya doc_id
            retrieved_doc_ids = [r['doc_id'] for r in retrieved_docs[:k]]
            
            ap = average_precision(retrieved_doc_ids, relevant_docs)
            aps.append(ap)
            
    return np.mean(aps) if aps else 0.0
