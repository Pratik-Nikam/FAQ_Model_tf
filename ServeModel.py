from flask import Flask, request, jsonify, render_template
import random

app = Flask(__name__)

# Basic message processing function
def process_message(message):
    responses = [
        "That's interesting!",
        "Tell me more.",
        "I see what you mean.",
        "Can you explain further?",
        "That sounds great!"
    ]
    return random.choice(responses)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get("message", "")
    
    if not user_message:
        return jsonify({"error": "Empty message"}), 400
    
    response_text = process_message(user_message)
    
    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True)


'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aware Testing Bot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f4f4f9;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }

        .chat-container {
            width: 400px;
            background: white;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            border-radius: 10px;
            overflow: hidden;
        }

        .chat-header {
            background: #007bff;
            color: white;
            text-align: center;
            padding: 15px;
            font-size: 18px;
            font-weight: bold;
        }

        .chat-note {
            background: #e9f5ff;
            padding: 10px;
            font-size: 14px;
            color: #333;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }

        .chat-box {
            height: 400px;
            overflow-y: auto;
            padding: 15px;
            border-bottom: 1px solid #ddd;
        }

        .message {
            margin: 10px 0;
            padding: 10px;
            border-radius: 5px;
            max-width: 75%;
        }

        .user-message {
            background: #007bff;
            color: white;
            text-align: right;
            align-self: flex-end;
            margin-left: auto;
        }

        .bot-message {
            background: #f1f1f1;
            color: black;
            text-align: left;
            align-self: flex-start;
        }

        .input-area {
            display: flex;
            padding: 10px;
            background: #fff;
        }

        .input-area input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 16px;
        }

        .input-area button {
            padding: 10px 15px;
            background: #007bff;
            color: white;
            border: none;
            border-radius: 5px;
            margin-left: 10px;
            cursor: pointer;
            font-size: 16px;
        }

        .input-area button:hover {
            background: #0056b3;
        }

    </style>
</head>
<body>

    <div class="chat-container">
        <div class="chat-header">Aware Testing Bot</div>
        <div class="chat-note">Provide the bot with just description.</div>
        <div class="chat-box" id="chatMessages"></div>
        <div class="input-area">
            <input type="text" id="userInput" placeholder="Type a message..." />
            <button onclick="sendMessage()">Send</button>
        </div>
    </div>

    <script>
        function sendMessage() {
            let userInput = document.getElementById("userInput").value;
            if (!userInput.trim()) return;

            let chatMessages = document.getElementById("chatMessages");
            chatMessages.innerHTML += `<div class="message user-message"><strong>You:</strong> ${userInput}</div>`;

            fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userInput })
            })
            .then(response => response.json())
            .then(data => {
                chatMessages.innerHTML += `<div class="message bot-message"><strong>Bot:</strong> ${data.response}</div>`;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });

            document.getElementById("userInput").value = "";
        }
    </script>

</body>
</html>
'''
