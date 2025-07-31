FROM rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.7.1
COPY src/clipora ./clipora
WORKDIR /clipora
RUN sudo apt-get install -y yq
RUN pip install --no-cache-dir -r amd_requirements.txt
