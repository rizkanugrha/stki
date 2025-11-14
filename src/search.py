import argparse
import os
import json
import sys

# Tambahkan path root (UTS) agar bisa import src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.boolean_ir import BooleanRetrieval
from src.vsm_ir import VectorSpaceModel

def main():
    # (Argumen CLI)
    parser = argparse.ArgumentParser(description="STKI Search Engine Orchestrator")
    parser.add_argument('--model', choices=['boolean', 'vsm'], required=True,
                        help="Model retrieval yang akan digunakan (boolean atau vsm).")
    parser.add_argument('--k', type=int, default=3,
                        help="Jumlah top-k dokumen yang ingin ditampilkan (hanya untuk VSM).")
    parser.add_argument('--query', type=str, required=True,
                        help="Query pencarian.")
    
    args = parser.parse_args()
    
    # Tentukan path (asumsi dijalankan dari root 'UTS')
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    
    if not os.path.exists(processed_dir):
        print(f"Error: Direktori '{processed_dir}' tidak ditemukan.")
        return

    if args.model == 'boolean':
        print(f"--- Menjalankan Boolean Model ---")
        print(f"Query: {args.query}")
        
        bool_model = BooleanRetrieval(processed_dir)
        results, explain = bool_model.process_query(args.query)
        
        print("\nPenjelasan Query:")
        print(explain)
        print(f"\nHasil Ditemukan: {len(results)} dokumen")
        print(results)
            
    elif args.model == 'vsm':
        print(f"--- Menjalankan Vector Space Model (VSM) ---")
        print(f"Query: {args.query}")
        
        vsm_model = VectorSpaceModel(raw_dir)
        results = vsm_model.search(args.query, k=args.k)
        
        print(f"\nTop-{args.k} Hasil (skor > 0.01):")
        if not results:
            print("Tidak ada dokumen yang relevan ditemukan.")
            return
            
        # (Keluaran: daftar dokumen + skor + explain singkat)
        for res in results:
            print(f"\n[{res['doc_id']}] - Skor Cosine: {res['score']:.4f}")
            print(f"   Snippet: {res['snippet']}")
            
if __name__ == "__main__":
    main()
