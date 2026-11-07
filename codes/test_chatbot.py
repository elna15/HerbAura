import os
from database_chatbot import DatabaseChatbot

current_dir = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(current_dir, "herbaura_database")

chatbot = DatabaseChatbot(db_path)

print("\nHerbAura Hybrid Chatbot Ready!")
print("Type 'bye' (or 'exit') to quit.\n")

quit_commands = {"bye", "exit", "quit", "goodbye"}

while True:
    query = input("You: ").strip()

    if query.lower() in quit_commands:
        print("Bot: Goodbye! Stay healthy!")
        break

    response = chatbot.get_response(query)
    print("\nBot:", response)
    print()
