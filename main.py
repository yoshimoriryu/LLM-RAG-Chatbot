from sqlalchemy import create_engine, inspect, text
import pandas as pd
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import re

load_dotenv()

DATABASE_URI = os.getenv('DATABASE_URI')
engine = create_engine(DATABASE_URI)

def extract_schema_and_enums(engine):
    inspector = inspect(engine)
    schema_info = []

    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            schema_info.append(f"Table: {table_name}")
            columns = inspector.get_columns(table_name)
            for col in columns:
                col_name = col['name']
                col_type = str(col['type'])
                schema_info.append(f" - {col_name} ({col_type})")

                # Attempt enum-like value discovery for text-based fields
                if any(t in col_type.lower() for t in ["char", "text"]):
                    try:
                        enum_query = text(f"""
                            SELECT {col_name}, COUNT(*) as freq
                            FROM {table_name}
                            WHERE {col_name} IS NOT NULL
                            GROUP BY {col_name}
                            ORDER BY freq DESC
                            LIMIT 10;
                        """)
                        result = conn.execute(enum_query).fetchall()
                        enum_values = [row[0] for row in result if row[0] is not None]
                        if 1 <= len(enum_values) <= 10:
                            schema_info.append(f"   * Possible values for {col_name}: {enum_values}")
                    except Exception as e:
                        pass  # skip if column can't be queried this way

            schema_info.append("")  # spacing between tables

    return "\n".join(schema_info)

def get_db_schema(engine):
    query = """
    SELECT table_name, column_name
    FROM information_schema.columns
    WHERE table_schema = 'public'
    ORDER BY table_name, ordinal_position;
    """
    with engine.connect() as conn:
        result = conn.execute(text(query))
        rows = result.fetchall()

    from collections import defaultdict
    schema = defaultdict(list)
    for table, column in rows:
        schema[table].append(column)

    # Format into a string block
    schema_str = "\n".join([f"- {table}({', '.join(columns)})" for table, columns in schema.items()])
    return schema_str

schema_str = extract_schema_and_enums(engine)

print('This is schema_str')
print(schema_str)

router_prompt = ChatPromptTemplate.from_template(f"""
                                                 
Database Schema:
{schema_str}

==========

You are a strict classifier that routes user questions to task types. I give you database schema as context.

You must classify the question into exactly ONE of the following task type:
- SQL_QUERY
- DATA_EXPLAIN
- OTHER

Respond with ONLY the task type. Answer with one word.
Question: {{input}}
""")

router_llm = Ollama(model="llama3:8b") # sensitive, gemma:2b failed to give only 1 answer
router_chain = router_prompt | router_llm

# Agent 2: SQL Executor
def run_sql_query(engine, question):
    print("Agent SQL contacted.")
    def clean_sql(raw_sql: str) -> str:
        # Remove triple backticks and leading 'sql' (if present)
        return re.sub(r"```sql|```", "", raw_sql, flags=re.IGNORECASE).strip()

    # Use LLM to generate SQL
    sql_gen_llm = Ollama(model="gemma:2b")
    sql_prompt = f"""
    You are an expert SQL generator.

    Here is the database schema:
    {schema_str}

    Write a PostgreSQL query to answer the user's question.
    Only return the SQL. No explanations.

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

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() in ["exit", "quit"]:
            break

        try:
            route = router_chain.invoke({"input": question})
            route = route.strip().upper()
            print("\n[Router decided task is]", route)
            print(route, route == "SQL_QUERY")
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
