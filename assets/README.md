# Assets

Đặt hai portrait gốc vào thư mục này (không đổi tên, không chỉnh sửa):

```text
assets/
├── tanvu.jpeg    <- copy từ /mnt/data/tanvu.jpeg
├── himass.jpeg   <- copy từ /mnt/data/himass.jpeg
└── README.md     <- file này
```

## Cách copy (môi trường có /mnt/data)

```bash
cp /mnt/data/tanvu.jpeg assets/tanvu.jpeg
cp /mnt/data/himass.jpeg assets/himass.jpeg
```

## Windows (nếu đã tải file về máy)

Copy thủ công `tanvu.jpeg` và `himass.jpeg` vào thư mục `assets/`.

## Fallback khi thiếu file

UI (`app.py`) tự phát hiện file còn thiếu và hiển thị khối placeholder
tĩnh (chữ TANVUU / HIMASS) thay cho ảnh — chat vẫn hoạt động bình thường.
Không commit ảnh thay thế do AI sinh ra; chỉ dùng đúng hai portrait gốc.
