from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image, ImageDraw
import torch, os, glob
from scipy.spatial.distance import cosine
import numpy as np
from sklearn.preprocessing import normalize

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn = MTCNN(image_size=160, margin=0, keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# --- Helper: get embedding for a single image ---
def get_embedding(img_path):
    img = Image.open(img_path).convert("RGB")
    faces = mtcnn(img)
    if faces is None:
        raise ValueError(f"No face detected in {img_path}")
    emb = resnet(faces[0].unsqueeze(0).to(device)).detach().cpu().numpy()
    emb = normalize(emb)[0]  # L2 normalize
    return emb

# --- Helper: average embeddings from multiple images ---
def get_avg_embedding(img_paths):
    embs = []
    for p in img_paths:
        try:
            embs.append(get_embedding(p))
            print(f"Processed {os.path.basename(p)}")
        except Exception as e:
            print(f"⚠️ Skipped {p}: {e}")
    if not embs:
        raise ValueError("No valid embeddings found for reference images")
    return np.mean(embs, axis=0)

# --- 1. Load John's reference embeddings ---
reference_dir = r"C:\programming\facenet\my_faces"
reference_paths = glob.glob(os.path.join(reference_dir, "john*.jpg")) + \
                  glob.glob(os.path.join(reference_dir, "john*.png"))

john_embedding = get_avg_embedding(reference_paths)
print("✅ Loaded reference embedding for John")

# --- 2. Load a crowd image ---
crowd_path = r"C:\programming\facenet\my_faces\crowd.jpg"  # change as needed
crowd_img = Image.open(crowd_path).convert("RGB")

# --- 3. Detect all faces in crowd ---
boxes, probs = mtcnn.detect(crowd_img)

if boxes is None:
    print("No faces detected in crowd image.")
    exit()

# --- 4. Compute embeddings for each detected face ---
faces = mtcnn(crowd_img)
embeddings = resnet(faces.to(device)).detach().cpu().numpy()
embeddings = normalize(embeddings)  # normalize all

# --- 5. Compare each embedding to John's ---
draw = ImageDraw.Draw(crowd_img)
threshold = 0.8  # stricter threshold

for i, (box, emb, prob) in enumerate(zip(boxes, embeddings, probs)):
    dist = cosine(john_embedding, emb)
    similarity = 1 - dist

    if dist < threshold:
        color = (0, 255, 0)  # green for match
        label = f"John (d={dist:.2f})"
    else:
        color = (255, 0, 0)  # red for not John
        label = f"Other (d={dist:.2f})"

    # Draw box + label
    draw.rectangle(box.tolist(), outline=color, width=3)
    draw.text((box[0], box[1]-10), label, fill=color)

# --- 6. Show result ---
crowd_img.show()
output_path = os.path.splitext(crowd_path)[0] + "_marked.jpg"
crowd_img.save(output_path)
print(f"✅ Result saved to {output_path}")
