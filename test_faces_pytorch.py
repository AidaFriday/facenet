from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch, glob, os
from scipy.spatial.distance import cosine

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Helper: get embedding from image ---
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    face = mtcnn(img)
    if face is None:
        raise ValueError(f"No face detected in {img_path}")
    emb = resnet(face.unsqueeze(0).to(device))
    return emb.detach().cpu().numpy()[0]

# --- Load all images in my_faces/ ---
img_dir = "my_faces"
embeddings = {}
for ext in ("*.jpg", "*.png"):
    for path in glob.glob(os.path.join(img_dir, ext)):
        name = os.path.splitext(os.path.basename(path))[0]
        embeddings[name] = get_embedding(path)
        print(f"Processed {name}")

# --- Compare distances ---
names = list(embeddings.keys())
for i in range(len(names)):
    for j in range(i+1, len(names)):
        d = cosine(embeddings[names[i]], embeddings[names[j]])
        print(f"Similarity {names[i]} vs {names[j]}: {1-d:.4f}, distance={d:.4f}")
