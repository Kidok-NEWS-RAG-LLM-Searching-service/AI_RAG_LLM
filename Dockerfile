FROM python:3.12
COPY ./.env.dev /code/.env.dev

WORKDIR /code
COPY ./requirements.txt /app/requirements.txt

RUN pip install -r /app/requirements.txt

EXPOSE 8000

# Copy the env file into the image

# 애플리케이션 코드 복사
COPY ./app /code/app

ARG ENV=dev
ENV ENV=${ENV}

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

# CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port 80 ENV=dev"]
