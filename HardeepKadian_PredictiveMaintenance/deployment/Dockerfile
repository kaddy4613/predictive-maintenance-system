# ==========================================
# BASE IMAGE
# ==========================================

FROM python:3.10

# ==========================================
# WORKING DIRECTORY
# ==========================================

WORKDIR /app

# ==========================================
# COPY FILES
# ==========================================

COPY . .

# ==========================================
# INSTALL DEPENDENCIES
# ==========================================

RUN pip install --no-cache-dir -r requirements.txt

# ==========================================
# CREATE NON-ROOT USER
# ==========================================

RUN useradd -m -u 1000 user

USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# ==========================================
# STREAMLIT CONFIG
# ==========================================

EXPOSE 8501

# ==========================================
# START APPLICATION
# ==========================================

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
