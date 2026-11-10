FROM python:3.11-slim

WORKDIR /app

# 先装依赖层，利用镜像缓存；源码改动只重建最后两层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 7860
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]