"""Module factice. fkd_diffusers/llm_grading.py fait `from google import genai` a l'import ;
le grader LLM n'est jamais appele avec ImageReward. Pas de __init__.py dans shim/google/ :
le paquet `google` reste un namespace, celui de protobuf compris."""


class Client:  # jamais instancie
    def __init__(self, *a, **k):
        raise RuntimeError("google.genai factice : le grader LLM n'est pas disponible ici")


types = None
