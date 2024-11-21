from TravelChatBot import TravelChatBot


if __name__ == "__main__":
    chatbot = TravelChatBot()
    print(chatbot.greet())

    while True:
        message = input("Human: ")
        if message.lower() == "quit":
            break
        print("Bot:", chatbot.respond(message))
