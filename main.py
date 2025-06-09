from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama

app = FastAPI()
templates = Jinja2Templates(directory="templates")

llm = Llama(model_path="./models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, user_input: str = Form(...)):
    output = llm(
        prompt=f"User: {user_input}\nAssistant:",
        max_tokens=256,
        stop=["User:", "Assistant:"]
    )
    resposta = output["choices"][0]["text"].strip()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": user_input,
        "response": resposta
    })
