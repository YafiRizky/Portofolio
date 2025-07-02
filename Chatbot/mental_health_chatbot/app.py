from flask import Flask, render_template, request, jsonify
from chatbot import MentalHealthChatbot
import json
import os

app = Flask(__name__)
chatbot = MentalHealthChatbot()

def load_structured_topics():
    """
    IMPROVEMENT: Memuat daftar topik terstruktur dari topics.json untuk UI.
    Ini menjadi satu-satunya sumber untuk daftar topik di frontend.
    """
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        with open(os.path.join(data_dir, 'topics.json'), 'r', encoding='utf-8') as f:
            structured_topics = json.load(f)
            return structured_topics.get("categories", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"ERROR: Gagal memuat topik dari topics.json: {e}")
        return []

@app.route("/")
def home():
    """Menampilkan halaman utama dengan daftar hotline dan topik terstruktur."""
    try:
        hotlines = chatbot.get_hotlines()
        # Menggunakan fungsi baru untuk memuat topik
        topic_categories = load_structured_topics()
        # Mengirim 'topic_categories' ke template
        return render_template("index.html", hotlines=hotlines, topic_categories=topic_categories)
    except Exception as e:
        print(f"Error in home route: {e}")
        return f"Error: {e}"

@app.route("/get_response", methods=["POST"])
def get_response():
    """Endpoint untuk mendapatkan respons dari chatbot."""
    try:
        user_input = request.form["user_input"]
        response = chatbot.generate_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        print(f"Error in get_response: {e}")
        return jsonify({"error": f"Terjadi kesalahan di server: {str(e)}"}), 500

if __name__ == "__main__":
    if not os.path.exists('templates'):
        print("ERROR: Direktori 'templates' tidak ditemukan.")
    if not os.path.exists('data'):
        print("ERROR: Direktori 'data' tidak ditemukan.")
    
    app.run(debug=True)
