FROM python:3.10.3-slim
EXPOSE 8080
COPY requirements.txt app/requirements.txt
RUN pip install --no-cache-dir -r app/requirements.txt
COPY . /app
WORKDIR /app
ENTRYPOINT ["streamlit", "run", "/forecasting/run_forecaster.py", "--server.port=8080", "--server.address=0.0.0.0"]