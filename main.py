from duckduckgo_search import ddg
from llama_cpp import Llama
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="templates")

llm = Llama(
    model_path="./models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    n_ctx=4096
)

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/ask", response_class=HTMLResponse)
def ask(request: Request, user_input: str = Form(...)):
    if user_input.lower().startswith("pesquise") or user_input.lower().startswith("search"):
        query = user_input.replace("pesquise", "").replace("search", "").strip()

        try:
            results = ddg(query, max_results=2)
            snippet = "\n".join([r["body"] for r in results if "body" in r])
            if not snippet:
                snippet = "Nenhum resultado encontrado na busca."
        except Exception as e:
            snippet = f"Erro ao buscar: {str(e)}"

        prompt = f"O usuário pediu para pesquisar: '{query}'. Aqui está o que foi encontrado:\n{snippet}\nExplique com suas palavras:"
    else:
        prompt = f"""You are a helpful assistant. Always respond in the same language that the user uses.

User: {user_input}
Assistant:"""

    output = llm(prompt=prompt, max_tokens=256, stop=["User:", "Assistant:"])
    resposta = output["choices"][0]["text"].strip()

    return templates.TemplateResponse("index.html", {
        "request": request,
        "user_input": user_input,
        "response": resposta
    })
