# deployment/main.py

from deployment.online.api import InferenceAPI

if __name__ == "__main__":
    InferenceAPI().serve()
