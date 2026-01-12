from ollama import ChatResponse, chat
from tasks import InstructionsFollowing

task_1 = InstructionsFollowing()

response_zero_shot: ChatResponse = chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "user",
            "content": task_1.prompt,
        },
    ],
)

response_few_shot: ChatResponse = chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "user",
            "content": task_1.few_shot_prompt,
        },
    ],
)

reponse_cot: ChatResponse = chat(
    model="qwen2.5:1.5b",
    messages=[
        {
            "role": "user",
            "content": task_1.cot_prompt,
        },
    ],
)

# print(response["message"]["content"])
# # or access fields directly from the response object
# print(response.message.content)

print(response_zero_shot.message.content)
print("-----")
print(response_few_shot.message.content)
print("-----")
print(reponse_cot.message.content)
