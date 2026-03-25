def build_prompt(messages):

    system_prompt = (
        "You are a helpful AI assistant. "
        "Answer clearly and concisely. "
        "Do not continue the conversation by writing User or Assistant roles. "
        "Only answer the user's latest question.\n\n"
    )

    conversation = ""

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user":
            conversation += f"User: {content}\n"
        else:
            conversation += f"Assistant: {content}\n"

    conversation += "Assistant:"

    return system_prompt + conversation
