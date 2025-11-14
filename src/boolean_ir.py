import os
from src.preprocess import preprocess

class BooleanRetrieval:
    def __init__(self, processed_dir):
        self.inverted_index = {}
        self.doc_contents = {}
        self.all_doc_ids = set()
        self._build_index(processed_dir)

    def _build_index(self, processed_dir):
        """Membangun inverted index dari dokumen yang sudah diproses."""
        # (build_inverted_index)
        doc_files = sorted([f for f in os.listdir(processed_dir) if f.endswith('.txt')])
        
        for doc_id in doc_files:
            self.all_doc_ids.add(doc_id)
            filepath = os.path.join(processed_dir, doc_id)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                self.doc_contents[doc_id] = content
                
                # Gunakan teks yang sudah diproses untuk index
                tokens = content.split()
                
                for token in set(tokens): # Gunakan set untuk term unik per dokumen
                    if token not in self.inverted_index:
                        self.inverted_index[token] = set()
                    self.inverted_index[token].add(doc_id)

    def _get_postings(self, term):
        """Mendapatkan postings list untuk satu term (sudah diproses)."""
        return self.inverted_index.get(term, set())

    def process_query(self, query):
        """Memproses query boolean sederhana (AND, OR, NOT)."""
        # (Parser query Boolean sederhana)
        processed_query = preprocess(query)
        tokens = processed_query.split()
        
        if not tokens:
            return set(), "Query kosong"
        
        # Logika parser sederhana (asumsi tidak ada tanda kurung)
        if tokens[0].upper() == 'NOT':
            if len(tokens) < 2: return set(), "Query NOT tidak valid"
            result_set = self.all_doc_ids - self._get_postings(tokens[1])
            start_index = 2
            explain_log = f"NOT {tokens[1]} ({len(result_set)} dok)"
        else:
            result_set = self._get_postings(tokens[0])
            start_index = 1
            explain_log = f"{tokens[0]} ({len(result_set)} dok)"

        
        i = start_index
        while i < len(tokens):
            op = 'AND' # Operator default jika tidak ada
            if tokens[i].upper() in ['AND', 'OR']:
                op = tokens[i].upper()
                i += 1
                if i >= len(tokens):
                    return set(), "Query tidak valid (operator di akhir)"
            
            term = tokens[i]
            
            is_not = False
            if term.upper() == 'NOT':
                is_not = True
                i += 1
                if i >= len(tokens):
                    return set(), "Query tidak valid (NOT di akhir)"
                term = tokens[i]
                
            term_postings = self._get_postings(term)
            if is_not:
                term_postings = self.all_doc_ids - term_postings
                explain_log += f" {op} (NOT {term}) "
            else:
                explain_log += f" {op} {term} "

            if op == 'AND':
                result_set = result_set.intersection(term_postings) # 
            elif op == 'OR':
                result_set = result_set.union(term_postings) # 
            
            explain_log += f"-> {len(result_set)} dok"
            i += 1
            
        return sorted(list(result_set)), explain_log
