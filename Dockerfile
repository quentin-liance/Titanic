FROM python:3.7.5-slim
 
ENV PYTHONUNBUFFERED=TRUE
 
RUN pip --no-cache-dir install pipenv
 
WORKDIR /app
 
COPY ["Pipfile", "Pipfile.lock", "./"]
 
RUN pipenv install --deploy --system && \
    rm -rf /root/.cache
 
COPY ["*.py", "model_file.p", "./"]
 
EXPOSE 9696
 
ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "titanic_serving:app"]
