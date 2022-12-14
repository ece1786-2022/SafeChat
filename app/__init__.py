from flask import Flask
from app.secure import OPENAI_KEY, HUGGINGFACE_KEY


webapp = Flask(__name__)
webapp.config.update(
    DEBUG=True,
    SECRET_KEY = 'this-is-a-secret-key',
    OPENAI_KEY = OPENAI_KEY,
    HUGGINGFACE_KEY = HUGGINGFACE_KEY
)

from app import main
