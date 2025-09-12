# load json
# per sentence, 
from sentence-transformer import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

list_of_words

embeddings = model.encode(list_of_words, show_progress_bar=True)
