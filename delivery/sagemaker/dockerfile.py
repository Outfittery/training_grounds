DOCKERFILE_TEMPLATE = '''FROM python:{python_version}

{install_libraries}

RUN pip install sagemaker-containers 

COPY . /opt/ml/code

WORKDIR /opt/ml/code

COPY {package_filename} package.tar.gz

RUN pip install package.tar.gz

WORKDIR /

# Defines train.py as script entrypoint
ENV SAGEMAKER_PROGRAM {run_file_name}
'''
