FROM pytorch/pytorch:2.9.0-cuda12.8-cudnn9-runtime

ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN pip install pyjuice==2.3.2 pandas==2.3.3 numpy==2.3.4 scikit-learn==1.7.2 scipy==1.16.3 spflow==0.0.41 seaborn matplotlib jupyterlab

RUN apt update && apt install -y build-essential git cmake curl unzip clangd

RUN curl -LO https://github.com/neovim/neovim/releases/latest/download/nvim-linux-x86_64.tar.gz && rm -rf /opt/nvim-linux-x86_64 && tar -C /opt -xzf nvim-linux-x86_64.tar.gz

ENV PATH="$PATH:/opt/nvim-linux-x86_64/bin"

RUN mkdir -p /root/.config/nvim
WORKDIR /root/.config/nvim
RUN git clone https://github.com/nvim-lua/kickstart.nvim.git .

ENV XDG_CONFIG_HOME=/root/.config

WORKDIR /app
RUN echo "jupyter lab --ip=0.0.0.0 --allow-root --port=8888 --no-browser --NotebookApp.token='' --NotebookApp.password=''" > jupyterlab
RUN chmod 777 jupyterlab

EXPOSE 8888

CMD ["bash"]
