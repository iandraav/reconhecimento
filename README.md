import cv2
import numpy as np
from mtcnn.mtcnn import MTCNN
from keras_facenet import FaceNet
from sklearn.preprocessing import LabelEncoder, Normalizer
from scipy.spatial.distance import cosine

# --- CARREGAR DADOS SALVOS ---
data = np.load('faces_embeddings.npz')
known_embeddings = data['embeddings']
known_labels_encoded = data['labels']

# Recria o LabelEncoder com as classes conhecidas
# (Uma forma mais robusta seria salvar o encoder com pickle)
all_labels = sorted(list(set(os.listdir('dataset/')))) 
label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

# Normaliza os embeddings conhecidos (melhora a comparação por distância)
in_encoder = Normalizer(norm='l2')
known_embeddings = in_encoder.transform(known_embeddings)

# --- INICIALIZAR MODELOS ---
detector = MTCNN()
embedder = FaceNet()

# --- INICIAR WEBCAM ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Converte para RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detecta faces no frame
    results = detector.detect_faces(frame_rgb)

    for result in results:
        x1, y1, width, height = result['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        
        # Extrai o rosto
        face_pixels = frame_rgb[y1:y2, x1:x2]
        
        if face_pixels.size == 0:
            continue
            
        # Prepara para a FaceNet
        face_pixels = cv2.resize(face_pixels, (160, 160))
        face_pixels = face_pixels.astype('float32')
        mean, std = face_pixels.mean(), face_pixels.std()
        face_pixels = (face_pixels - mean) / std
        face_array = np.expand_dims(face_pixels, axis=0)
        
        # Gera o embedding do rosto atual
        current_embedding = embedder.embeddings(face_array)
        current_embedding = in_encoder.transform(current_embedding)

        # --- COMPARAÇÃO ---
        min_dist = float('inf')
        best_match_idx = -1

        for i, known_emb in enumerate(known_embeddings):
            # Usando distância de cosseno (ótima para vetores de alta dimensão)
            dist = cosine(known_emb, current_embedding[0])
            if dist < min_dist:
                min_dist = dist
                best_match_idx = i

        # Define um limiar de reconhecimento
        # Este valor pode precisar de ajuste!
        recognition_threshold = 0.5 

        if min_dist <= recognition_threshold:
            # Reconheceu! Pega o nome da pessoa
            person_name = label_encoder.inverse_transform([known_labels_encoded[best_match_idx]])[0]
            color = (0, 255, 0) # Verde
        else:
            person_name = "Desconhecido"
            color = (0, 0, 255) # Vermelho
        
        # --- DESENHA NA TELA ---
        # Caixa delimitadora
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Texto com o nome
        text = f"{person_name} ({min_dist:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Mostra o resultado
    cv2.imshow('Reconhecimento Facial', frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
cap.release()
cv2.destroyAllWindows()
