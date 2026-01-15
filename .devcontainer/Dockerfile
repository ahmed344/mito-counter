FROM pytorch/pytorch:2.9.1-cuda13.0-cudnn9-runtime

# Install basic utilities
RUN apt-get update
RUN apt-get install -y git sudo curl libfontconfig1 libxrender1 libgl1-mesa-glx libglib2.0-0 \
    libx11-xcb1 libxkbcommon-x11-0 libxcb1 libxcb-icccm4 libxcb-image0 \
    libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xfixes0 \
    libxcb-xinerama0 libxcb-shape0 libxkbcommon0 libdbus-1-3

# Add user so we don't need to run VS Code as root
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /usr/bin/bash \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# [Optional] Set the default user. Omit if you want to keep the default as root.
USER $USERNAME

# Copy requirements.txt and install dependencies
COPY requirements.txt .
RUN pip install --upgrade --no-cache-dir -r requirements.txt