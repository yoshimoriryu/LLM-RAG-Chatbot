from sqlalchemy import create_engine, text
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URI = os.getenv('DATABASE_URI')

def run_sql_query(engine, question):
    with engine.connect() as conn:
        if "largest balance" in question.lower():
            result = conn.execute(text("SELECT * FROM users ORDER BY balance DESC LIMIT 1"))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            return f"The row with the largest balance is: {df.to_dict(orient='records')[0]}"
        elif "average balance" in question.lower() and "older than" in question.lower():
            age = int("".join([c for c in question if c.isdigit()]))
            result = conn.execute(text(f"SELECT AVG(balance) FROM users WHERE age > {age}"))
            avg = result.scalar()
            return f"The average balance for users older than {age} is {avg:.2f}"
        elif "average balance" in question.lower():
            result = conn.execute(text("SELECT AVG(balance) FROM users"))
            avg = result.scalar()
            return f"The average balance is {avg:.2f}"
        else:
            return "Sorry, I don't know how to answer that yet."

def main():
    # Connect to PostgreSQL
    print(DATABASE_URI)
    engine = create_engine(DATABASE_URI)

    # Initialize LLM
    from langchain_community.llms import Ollama
    llm = Ollama(model="gemma:2b")

    while True:
        question = input("Ask a question (or type 'exit' to quit): ")
        if question.lower() in ["exit", "quit"]:
            break
        try:
            context = run_sql_query(engine, question)
            print("Answer:", llm.invoke(f"Answer the question based on this data:\n{context}\n\nQ: {question}"))
        except Exception as e:
            print("Error:", e)


if __name__ == "__main__":
    main()