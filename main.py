import sys

from llm_sdk import Small_LLM_Model


def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py '<question>'")
        return

    model = Small_LLM_Model()

    # Convertimos la pregunta en IDs
    input_ids = model.encode(sys.argv[1]).tolist()[0]

    # Generamos 30 tokens
    generated_ids = []

    for _ in range(30):
        logits = model.get_logits_from_input_ids(input_ids)

        # Elegimos el token con mayor puntuación
        next_id = max(range(len(logits)), key=logits.__getitem__)

        # Actualizamos el contexto
        input_ids.append(next_id)
        generated_ids.append(next_id)

    # Convertimos los tokens generados a texto
    print(model.decode(generated_ids))


if __name__ == "__main__":
    main()
