DOCKERFILE_TEMPLATE = '''FROM python:{python_version}

{install_libraries}

COPY . /opt/ml/code

COPY entry.py /opt/ml/code/sagemaker_executing_app/
COPY entry.pkl /opt/ml/code/sagemaker_executing_app/

WORKDIR /opt/ml/code

RUN pip install -e .

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM sagemaker_executing_app.entry
'''
