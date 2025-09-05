from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import glob, os

# Setup MTCNN detector
mtcnn = MTCNN(keep_all=True)

# Path to your images
img_dir = "my_faces"

# Loop through all images
for ext in ("*.jpg", "*.png"):
    for path in glob.glob(os.path.join(img_dir, ext)):
        img = Image.open(path).convert("RGB")
        
        # Detect faces
        boxes, probs, landmarks = mtcnn.detect(img, landmarks=True)
        
        # Draw results
        img_draw = img.copy()
        draw = ImageDraw.Draw(img_draw)

        if boxes is not None:
            for box, lm in zip(boxes, landmarks):
                draw.rectangle(box.tolist(), outline=(0,255,0), width=3)
                for point in lm:
                    draw.ellipse((point[0]-2, point[1]-2, point[0]+2, point[1]+2), fill=(255,0,0))
        
        # Show image
        img_draw.show()
        print(f"Processed {os.path.basename(path)}")
