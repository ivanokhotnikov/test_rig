FROM python:3.10.3-slim
EXPOSE 8081
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["streamlit", "run", "run_forecaster.py", "--server.port=8081", "--server.address=0.0.0.0"]