import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis
app = FaceAnalysis(name="buffalo_sc", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0)

known_names = []
known_embeddings = []

for file in os.listdir("ex_faces"):
    path = os.path.join("ex_faces", file)
    name = os.path.splitext(file)[0]
    img = cv2.imread(path)
    faces = app.get(img)
    if len(faces) > 0:
        known_names.append(name)
        known_embeddings.append(faces[0].embedding)

known_embeddings = np.array(known_embeddings, dtype=np.float32)

np.savez("known_faces.npz", names=known_names, embeddings=known_embeddings)
print("embeddings in known_faces.npz")
