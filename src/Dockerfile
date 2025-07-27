FROM python:3.12.3

# Set the working directory
WORKDIR /src

COPY . .

RUN mkdir -p /Data && chmod -R 777 /Data

ENV PYTHONPATH=/src

RUN pip install --no-cache-dir fastapi uvicorn pydantic

RUN pip install --no-cache-dir -r requirements_idx.txt

EXPOSE 93

CMD ["uvicorn", "main:mind", "--host", "0.0.0.0", "--port", "93"]