from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import re

load_dotenv()

DATABASE_URI = os.getenv('DATABASE_URI')

# Agent 1: Task Router
router_prompt = ChatPromptTemplate.from_template("""
You are a task routing assistant.

Your job is to classify the user's question into one of these exact task types:
- SQL_QUERY
- DATA_EXPLAIN
- OTHER

Respond with only the task type label. Do not include any explanation.

Strictly respond with ONE of:
SQL_QUERY
DATA_EXPLAIN
OTHER

Question: {input}

Task Type:
""")


router_llm = Ollama(model="gemma:2b")
router_chain = router_prompt | router_llm

# Agent 2: SQL Executor
def run_sql_query(engine, question):
    def clean_sql(raw_sql: str) -> str:
        # Remove triple backticks and leading 'sql' (if present)
        return re.sub(r"```sql|```", "", raw_sql, flags=re.IGNORECASE).strip()

    # Use LLM to generate SQL
    sql_gen_llm = Ollama(model="gemma:2b")
    sql_prompt = f"""
You are an expert SQL generator.

Based on the user's question, write a PostgreSQL query that can be used to answer it.
Only generate the SQL query, no explanation or formatting.

User Question:
{question}

PostgreSQL Query:
"""
    generated_sql = sql_gen_llm.invoke(sql_prompt).strip()

    # Print or log generated SQL
    print("[Generated SQL]:", generated_sql)
    sql = clean_sql(generated_sql)
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return df.to_string(index=False)
    except Exception as e:
        return f"[SQL ERROR]: {e}"


# Agent 3: Data Explainer
def explain_data(context, question):
    explainer_llm = Ollama(model="gemma:2b")
    prompt = f"""
You are a data analyst assistant. Your job is to interpret the *result* of a data query in plain language for non-technical users.

Do not explain the query itself — explain what the *result* means.

Here is the result:
{context}

Here is what the user asked:
{question}

Now explain the result in a helpful, clear way:
"""
    return explainer_llm.invoke(prompt)


def main():
    engine = create_engine(DATABASE_URI)

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() in ["exit", "quit"]:
            break

        try:
            route = router_chain.invoke({"input": question})
            route = route.strip().upper()
            print("\n[Router decided task is]", route)

            if route == "SQL_QUERY":
                context = run_sql_query(engine, question)
                answer = explain_data(context, question)
                print("Answer:", answer)

            elif route == "DATA_EXPLAIN":
                answer = explain_data("(no DB query needed)", question)
                print("Answer:", answer)

            else:
                print("Answer:", "Sorry, I cannot handle that request.")

        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()
