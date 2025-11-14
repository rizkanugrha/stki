import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# --- Inisialisasi ---
factory = StemmerFactory()
stemmer = factory.create_stemmer()
stop_words = set(stopwords.words('indonesian'))

def clean_text(text):
    """Membersihkan teks: lowercase, hapus URL, angka, dan tanda baca."""
    # (Case-folding, normalisasi angka/tanda baca)
    text = text.lower()  # Case-folding
    text = re.sub(r'https?://\S+', '', text)  # Hapus URL
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation)) # Hapus tanda baca
    text = text.strip()  # Hapus whitespace
    return text

def tokenize(text):
    """Tokenisasi teks."""
    # (Tokenisasi)
    return word_tokenize(text)

def remove_stopwords(tokens):
    """Menghapus stopwords dari list token."""
    # (Stopword removal)
    return [word for word in tokens if word not in stop_words and len(word) > 1]

def stem(tokens):
    """Melakukan stemming pada list token."""
    # (Stemming)
    return [stemmer.stem(word) for word in tokens]

def preprocess(text):
    """Pipeline preprocessing lengkap."""
    cleaned_text = clean_text(text)
    tokens = tokenize(cleaned_text)
    stopped_tokens = remove_stopwords(tokens)
    stemmed_tokens = stem(stopped_tokens)
    # Mengembalikan sebagai string yang dipisahkan spasi
    return " ".join(stemmed_tokens)
