from llama_cpp import Llama

llm = Llama(model_path="./models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf")

print("AquilesGPT pronto! Digite 'sair' para encerrar.\n")

history = []

while True:
    user_input = input("Você: ")

    if user_input.lower() in ["sair", "exit", "quit"]:
        break

    history.append({"role": "user", "content": user_input})

    prompt = "\n".join([f"{x['role'].capitalize()}: {x['content']}" for x in history])

    output = llm(prompt=f"{prompt}\nAssistant:", max_tokens=256, stop=["\nUser:", "\nVocê:"])

    resposta = output["choices"][0]["text"].strip()

    print(f"AquilesGPT: {resposta}\n")

    history.append({"role": "assistant", "content": resposta})
