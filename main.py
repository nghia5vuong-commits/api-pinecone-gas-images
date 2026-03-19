from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pinecone import Pinecone
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import io

app = FastAPI()

# Cho phép Google Apps Script gọi API này mà không bị lỗi CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# 1. KHỞI TẠO PINECONE
# ==========================================
# Thay bằng API Key thực tế của bạn
PINECONE_API_KEY = "pcsk_8m4tu_RxjD7VaU9sPwac76jKSPsu5aQpCp8CFfVU7rGeabz7BGjknFTpB7NFqiq9wuG3U"
pc = Pinecone(api_key=PINECONE_API_KEY)

# Tên Index bạn vừa tạo ở bước trước
index = pc.Index("drive-images") 

# ==========================================
# 2. TẢI MODEL AI (CLIP)
# ==========================================
print("Đang tải model CLIP... (Lần đầu sẽ hơi lâu)")
model_id = "openai/clip-vit-base-patch32" # Model xuất ra đúng 512 chiều
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)
print("Tải model thành công!")

# Hàm phụ: Biến ảnh thành Vector
def get_image_vector(image: Image.Image):
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    # Trả về mảng 512 số (chuyển tensor thành list Python)
    return outputs.detach().numpy()[0].tolist()

# ==========================================
# 3. TẠO ENDPOINT TÌM KIẾM CHO APPS SCRIPT
# ==========================================
@app.post("/api/search-image")
async def search_image(file: UploadFile = File(...)):
    try:
        # Đọc file ảnh từ web gửi lên
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        
        # Bước A: Nhờ AI biến ảnh thành Vector (512 số)
        query_vector = get_image_vector(image)
        
        # Bước B: Mang Vector này sang Pinecone hỏi xem 6 ảnh nào giống nhất
        response = index.query(
            vector=query_vector,
            top_k=6,
            include_metadata=True # Để lấy được cái File ID lưu kèm
        )
        
        # Bước C: Gói gọn kết quả trả về cho Apps Script
        results = []
        for match in response['matches']:
            results.append({
                "id": match['metadata']['fileId'], # ID của file trên Google Drive
                "score": match['score']            # Điểm giống nhau (Cosine: 1 là giống hệt)
            })
            
        return {"status": "success", "data": results}

    except Exception as e:
        return {"status": "error", "message": str(e)}

# Lệnh để chạy server (nếu bạn test trên máy tính local):
# uvicorn main:app --reload