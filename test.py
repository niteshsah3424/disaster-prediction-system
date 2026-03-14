from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)


history_data = []


while True:

    user_input = input("user input:")

    if user_input == "exit":
        break

    history_data.append({"role": "user", "content": user_input})

    completion = client.chat.completions.create(
        model="stepfun/step-3.5-flash:free",
        messages=history_data,
    )
    history_data.append(
        {"role": "assistant", "content": completion.choices[0].message.content}
    )

    print(completion.choices[0].message.content)