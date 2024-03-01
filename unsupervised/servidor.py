from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import pickle

app = Flask(__name__)

# Load the pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Load the pre-calculated embeddings for the text chunks
with open('embeddings_parrafos.pkl', 'rb') as embeddings_file:
    embeddings_parrafos = pickle.load(embeddings_file)

# Define trozos_de_texto globalmente
ruta_archivo = r"D:\UDEA\unsupervised\archivoprueba.txt"
with open(ruta_archivo, "r", encoding="utf-8") as archivo:
    texto = archivo.read()
    trozos_de_texto = texto.split("\n\n")  # Dividir por párrafos, ajustar según la estructura del texto

@app.route('/predict_relevant_chunks', methods=['POST'])
def predict_relevant_chunks():
    # Get the question from the JSON data in the request
    question = request.json['question']

    # Calculate the most relevant chunks for the question
    relevant_chunks = obtener_parralos_mas_relevantes(question, trozos_de_texto, embeddings_parrafos)

    # Return the relevant chunks as JSON response
    return jsonify({'relevant_chunks': relevant_chunks})

def obtener_parralos_mas_relevantes(question, trozos_de_texto, embeddings_parrafos, n=5):
    # Encode the question using the Sentence Transformer model
    embedding_question = model.encode([question], convert_to_tensor=True)
    
    # Calculate cosine similarity between the question and each text chunk
    similarities = cosine_similarity(embedding_question, embeddings_parrafos)[0]
    
    # Sort the text chunks based on their similarity to the question
    sorted_chunks = [chunk for _, chunk in sorted(zip(similarities, trozos_de_texto), reverse=True)]
    
    # Return the top N most relevant text chunks
    return sorted_chunks[:n]

if __name__ == '__main__':
    app.run(debug=True)


