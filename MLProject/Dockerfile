FROM python:3.10-slim
WORKDIR /app
COPY conda.yaml .
RUN pip install pandas numpy scikit-learn mlflow matplotlib
COPY modelling.py .
COPY MLProject .
COPY california_housing_preprocessing.csv .
ENV MLFLOW_TRACKING_URI=mlruns
CMD ["python", "modelling.py"]
