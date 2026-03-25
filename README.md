🤖 AI Bot (RAG + Agent System)

AI Bot is an intelligent agent-based system that combines Retrieval-Augmented Generation (RAG), vector search, and modular AI workflows to process user queries and generate context-aware responses. The system uses FAISS for vector storage and implements a graph-based agent architecture for dynamic decision-making.

🚀 Features
🤖 AI Agent with graph-based workflow
🔎 Retrieval-Augmented Generation (RAG)
📚 FAISS vector database for fast similarity search
🧠 Context-aware response generation
⚡ Modular node-based architecture
🔌 Extensible tools and routing system

🛠️ Tech Stack
Language: Python
AI Framework: Custom Agent System
Vector DB: FAISS
Libraries: NumPy, LangChain (if used), Requests
Architecture: Graph-based agent execution

📂 Project Structure
ai_bot/
│── ai_box/
│ ├── agent/
│ │ ├── graph.py # Workflow graph logic
│ │ ├── nodes.py # Agent nodes
│ │ ├── router.py # Routing logic
│ │ ├── tools.py # Tools integration
│ │
│ ├── rag/
│ │ ├── data.txt # Knowledge base
│ │ ├── vector_store.py # FAISS handling
│ │ └── faiss_index/ # Stored embeddings
│ │
│ ├── app.py # Main application
│ ├── model_loader.py # Model loading logic
│ ├── prompt.py # Prompt templates
│ ├── check_models.py # Model validation
│
│── test_llm.py # Testing script
│── requirements.txt # Dependencies
│── README.md # Documentation

⚙️ Installation
git clone https://github.com/meetkansara13/ai_bot.git
cd ai_bot
pip install -r requirements.txt

▶️ Usage
🔹 Run Application
python ai_box/app.py
🔹 Run Test Script
python test_llm.py

🧠 How It Works
User query is received
Agent router decides processing flow
RAG module retrieves relevant context using FAISS
Context + query is passed to model
Response is generated and returned

🔎 RAG Pipeline
Load knowledge base (data.txt)
Convert into embeddings
Store in FAISS index
Perform similarity search
Inject context into response generation

📌 Future Improvements
🌐 Web-based UI for interaction
🧠 Integration with advanced LLMs
📊 Conversation memory
🔐 Authentication & user sessions
☁️ Cloud deployment

👨‍💻 Contributor
Mrugesh Kansara (Meet) – Project Owner
