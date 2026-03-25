FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    curl \
    nano \
    unzip \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -s /bin/bash devuser

RUN curl -s https://ohmyposh.dev/install.sh | bash -s -- -d /usr/local/bin -t /usr/share/oh-my-posh/themes
RUN echo 'eval "$(oh-my-posh init bash --config /usr/share/oh-my-posh/themes/onehalf.minimal.omp.json)"' >> /home/devuser/.bashrc

RUN git clone --depth 1 https://github.com/junegunn/fzf.git /home/devuser/.fzf && \
    /home/devuser/.fzf/install --key-bindings --completion --no-update-rc && \
    chown -R devuser:devuser /home/devuser/.fzf
RUN echo '[ -f ~/.fzf.bash ] && source ~/.fzf.bash' >> /home/devuser/.bashrc

WORKDIR /workspace
RUN chown -R devuser:devuser /workspace

USER devuser
COPY requirements.txt .
RUN pip install --user -r requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu

