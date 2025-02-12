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
    <title>Flask Chat</title>
    <style>
        body { font-family: Arial, sans-serif; }
        .chat-box { width: 300px; margin: 20px auto; border: 1px solid #ccc; padding: 10px; }
        .messages { height: 200px; overflow-y: auto; border-bottom: 1px solid #ccc; margin-bottom: 10px; padding: 5px; }
        .input-area { display: flex; }
        .input-area input { flex: 1; padding: 5px; }
        .input-area button { padding: 5px; }
    </style>
</head>
<body>

    <div class="chat-box">
        <div class="messages" id="chatMessages"></div>
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
            chatMessages.innerHTML += `<div><strong>You:</strong> ${userInput}</div>`;

            fetch("/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message: userInput })
            })
            .then(response => response.json())
            .then(data => {
                chatMessages.innerHTML += `<div><strong>Bot:</strong> ${data.response}</div>`;
                chatMessages.scrollTop = chatMessages.scrollHeight;
            });

            document.getElementById("userInput").value = "";
        }
    </script>

</body>
</html>


'''
