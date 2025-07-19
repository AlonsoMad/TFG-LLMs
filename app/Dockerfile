FROM python:3.12.3

WORKDIR /app

COPY . .

ENV PYTHONPATH=/src

RUN pip install --no-cache-dir fastapi uvicorn pydantic

RUN pip install --no-cache-dir -r requirements_web.txt

EXPOSE 80

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
