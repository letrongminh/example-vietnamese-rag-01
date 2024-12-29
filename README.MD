## Vietnames RAG powered AI chatbot for an online flower shop ##

<img width="717" alt="Screen Shot 2024-09-26 at 12 50 05 AM" src="https://github.com/user-attachments/assets/cfb152d4-38a8-4c15-95b1-90ce6e6ab70d">

### Running in local ###
- Create an `.env` file and populate these ENV variables needed as required in  `.env.example`
- for `VECTOR_INDEX_NAME`, create a MongoDB Atlas vectorSearch index for the `DB_COLLECTION` collection:
```
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "embedding",
      "similarity": "cosine",
      "type": "vector"
    }
  ]
}
```
- for `KEYWORD_INDEX_NAME`, create a MongoDB Atlas search index for the `DB_COLLECTION` collection:
```
{
  "mappings": {
    "dynamic": false,
    "fields": {
      "title": [
        {
          "type": "string"
        }
      ]
    }
  }
}
```

- for `SEMANTIC_CACHE_INDEX_NAME`, create a MongoDB Atlas vectorSearch index for the `SEMANTIC_CACHE_COLLECTION` collection:
```
{
  "fields": [
    {
      "numDimensions": 768,
      "path": "embedding",
      "similarity": "euclidean",
      "type": "vector"
    },
    {
      "path": "llm_string",
      "type": "filter"
    }
  ]
}
```

- Install dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

- Run web scraper, `all_urls.json` and  `output.json` files will be generated:
```
python web_scraper.py
```

- Load documents and their embeddings into MongoDB:
```
python load_document.py
```

- Start the server, the server will run from http://localhost:5000 by default:
```
python serve.py
```

- Run streamlit app to access chatbot:
```
streamlit run Hello.py
```

- To run test suite:
```
python -m unittest
```


# Giải thích

Dưới đây là mô tả chi tiết về cách hoạt động của các thành phần trong dự án chatbot AI cho cửa hàng hoa trực tuyến.

## 1. Web Scraper (`web_scraper.py`)
**Chức năng:** Thu thập dữ liệu sản phẩm từ website hoa.

- **Phương pháp:** Sử dụng Scrapy để quét các URL từ `all_urls.json`.
- **Dữ liệu thu thập:** Thông tin sản phẩm như tiêu đề (`h1`), mô tả, giá cả, và URL hình ảnh.
- **Kết quả:** Lưu trữ dữ liệu vào `output.json`.

## 2. Tải Dữ liệu và Tạo Embedding (`load_document.py`)
**Chức năng:** Xử lý dữ liệu thu thập và tạo vector embeddings.

- **Quy trình:**
  - Đọc dữ liệu từ `output.json`.
  - Chia nhỏ nội dung thành các đoạn nhỏ dựa trên ngữ nghĩa.
  - Sử dụng SentenceTransformer để tạo embedding cho mỗi đoạn nội dung.
  - Lưu trữ dữ liệu đã xử lý vào MongoDB.

## 3. Mô Hình Embedding (`embedding_model/core.py`)
**Chức năng:** Tạo embedding cho văn bản.

- **Công cụ:** Sử dụng `SentenceTransformer` với mô hình `keepitreal/vietnamese-sbert`.
- **Phương pháp:** Mã hóa văn bản thành vector số để hỗ trợ tìm kiếm ngữ nghĩa.

## 4. Semantic Router (`semantic_router/router.py`)
**Chức năng:** Xác định liệu câu hỏi của người dùng thuộc về sản phẩm hay trò chuyện thông thường.

- **Phương pháp:**
  - Sử dụng embedding để tính toán độ tương đồng giữa câu hỏi và mẫu câu đã được định nghĩa.
  - Sắp xếp và chọn route phù hợp nhất dựa trên điểm số tương đồng.

## 5. RAG (`rag/core.py`)
**Chức năng:** Hỗ trợ tìm kiếm và trả lời thông tin dựa trên dữ liệu đã lưu trữ.

- **Quy trình:**
  - Thực hiện tìm kiếm kết hợp giữa vector search và keyword search.
  - Áp dụng thuật toán Weighted Reciprocal Rank để xếp hạng kết quả.
  - Tạo prompt được cải thiện với thông tin từ các tài liệu liên quan.

## 6. Semantic Cache (`semantic_cache/core.py`)
**Chức năng:** Lưu trữ và truy xuất các phản hồi đã được cache để tăng tốc độ phản hồi.

- **Phương pháp:**
  - Sử dụng vector search để tìm kiếm các phản hồi tương tự đã được lưu trữ.
  - Áp dụng ngưỡng điểm số để xác định cache hit.

## 7. Reflection (`reflection/core.py`)
**Chức năng:** Quản lý cuộc trò chuyện, ghi nhận lịch sử và xử lý phản hồi từ AI.

- **Chức năng chính:**
  - Xử lý tin nhắn từ người dùng và tạo phản hồi thông qua LLM (Large Language Model).
  - Ghi lại prompt từ người dùng và phản hồi từ AI vào MongoDB.
  - Lưu trữ phản hồi vào cache nếu cần thiết.

## 8. Server Flask (`serve.py`)
**Chức năng:** Cung cấp API để giao tiếp giữa giao diện người dùng và backend.

- **Endpoint chính:** `/api/v1/chat`
  - **Phương thức:** POST
  - **Quy trình:**
    - Nhận yêu cầu từ người dùng.
    - Sử dụng Semantic Router để định hướng câu hỏi.
    - Sử dụng RAG và Reflection để tạo phản hồi.
    - Trả về phản hồi dưới dạng JSON.

## 9. Giao Diện Streamlit (`pages/Chat Bot.py`)
**Chức năng:** Cung cấp giao diện người dùng để tương tác với chatbot.

- **Tính năng:**
  - Hiển thị logo và tiêu đề cửa hàng.
  - Quản lý trạng thái phiên trò chuyện.
  - Hiển thị lịch sử trò chuyện giữa người dùng và trợ lý AI.
  - Gửi yêu cầu tới API Flask và hiển thị phản hồi.

## 10. Các File Cấu Hình Khác
- **`.gitignore`:** Bỏ qua các thư mục và file không cần thiết như `__pycache__`, `venv/`, `.env`.
- **`README.MD`:** Hướng dẫn cài đặt và chạy dự án, bao gồm các bước cài đặt phụ thuộc, chạy web scraper, tải dữ liệu vào MongoDB, khởi động server và ứng dụng Streamlit.
- **`.env.example`:** Mẫu file môi trường chứa các biến cần thiết như `OPENAI_API_KEY`, `MONGO_URI`, `DB_NAME`, v.v.

## 11. Test Suite (`semantic_router/test_router.py`)
**Chức năng:** Kiểm thử các thành phần của Semantic Router để đảm bảo hoạt động đúng.

- **Phương pháp:** Sử dụng unittest để viết các trường hợp kiểm thử cho các chức năng của Semantic Router.

# Luồng Hoạt Động Tổng Thể

1. **Thu Thập Dữ Liệu:**
   - Chạy `web_scraper.py` để thu thập thông tin sản phẩm từ website và lưu vào `output.json`.

2. **Xử Lý và Tạo Embedding:**
   - Chạy `load_document.py` để xử lý dữ liệu, chia nhỏ nội dung, tạo embedding và lưu vào MongoDB.

3. **Khởi Động Server và Chatbot:**
   - Khởi động Flask server bằng cách chạy `serve.py`.
   - Khởi động ứng dụng giao diện người dùng bằng cách chạy `pages/Chat Bot.py` với Streamlit.

4. **Tương Tác Người Dùng:**
   - Người dùng tương tác qua giao diện Streamlit.
   - Yêu cầu được gửi tới API Flask.
   - Flask sử dụng Semantic Router để xác định loại câu hỏi.
   - Sử dụng RAG và Reflection để tạo phản hồi dựa trên dữ liệu đã lưu trữ.
   - Phản hồi được trả về và hiển thị trên giao diện người dùng.

5. **Cache và Tối Ưu:**
   - Các phản hồi có thể được lưu trữ trong Semantic Cache để tăng tốc độ trả lời trong các lần tương tác tiếp theo.

Qua các bước trên, hệ thống chatbot AI cung cấp trải nghiệm hỗ trợ khách hàng hiệu quả cho cửa hàng hoa trực tuyến, tận dụng mạnh mẽ các công nghệ như RAG, embedding models và caching để cải thiện chất lượng và tốc độ phản hồi.
