from langchain_core.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.runnables import RunnableMap

# Initialize local Gemma model from Ollama
llm = Ollama(model="gemma")

# Step 1: Prompt to extract intents
intent_prompt = PromptTemplate.from_template("""
You are a helpful assistant. Break this instruction into separate tasks, if there are more than one.
Instruction: {input}
Return only a numbered list of tasks.
""")

intent_chain = intent_prompt | llm

# Step 2: Task handlers (simple logic for now)
def handle_intent(task: str, doc: str):
    if "summarize" in task.lower():
        return llm.invoke(f"Summarize this: {doc}")
    elif "risk" in task.lower():
        return llm.invoke(f"Does this text mention any risks? Be specific: {doc}")
    else:
        return f"Unknown task: {task}"

# Full chain
def multi_intent_runner(user_input: str, doc: str):
    task_list_output = intent_chain.invoke({"input": user_input})
    
    print("\n🧠 Detected Tasks:\n", task_list_output)

    # Clean task list
    tasks = [line.strip()[3:] for line in task_list_output.split("\n") if line.strip().startswith("1.") or line.strip()[0].isdigit()]
    
    print("\n🔍 Handling Tasks:")
    results = []
    for task in tasks:
        print(f"- {task}")
        result = handle_intent(task, doc)
        results.append(f"{task}\n➡️ {result.strip()}\n")
    
    return "\n".join(results)

# === Run Demo ===
if __name__ == "__main__":
    user_query = "Summarize this paragraph and tell me if it mentions any risks."
    document = """
    Our Q2 financial report indicates stable revenue in most regions, but Southeast Asia faced
    operational delays due to regulatory issues. This may affect projected growth by 5% if not resolved.
    """

    response = multi_intent_runner(user_query, document)
    print("\n📝 Final Output:\n", response)
