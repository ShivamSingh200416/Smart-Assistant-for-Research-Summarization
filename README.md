# 🤖 Smart Assistant for Document Summarization & Q&A

A Streamlit-powered assistant that can:
- 🔍 Parse PDF or TXT files
- 📝 Generate intelligent summaries
- ❓ Answer user questions based on document content
- 🧠 Challenge users with logic-based questions and provide real-time evaluation

---

## 📦 Features
- **Ask Anything:** Ask free-form questions grounded in your uploaded document
- **Challenge Me:** Test your logical reasoning with context-aware question prompts
- **Citations & Justification:** See the paragraph your answer is based on
- **Streamlit UI:** Clean, interactive interface for seamless experimentation

---

## 🚀 Getting Started

### 1. Clone the Repo or Download the Project

If you’re cloning via Git:

```bash
git clone <your-repo-link>
cd ez-smart-assistant
```

Or manually unzip and navigate to the folder.

---

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

> This installs Streamlit, Transformers, PyPDF2, and other dependencies.

---

### 3. Launch the App

```bash
streamlit run app.py
```

Upload a PDF or TXT, and the assistant will summarize and analyze it!

---

## 🧪 Modules Overview

| File                  | Description |
|-----------------------|-------------|
| `app.py`              | Main Streamlit interface |
| `document_handler.py` | File parser (PDF/TXT) |
| `summarizer.py`       | Summarizes text with transformer |
| `qa_engine.py`        | Handles free-form questions |
| `logic_generator.py`  | Logic-based questioning & evaluation |
| `memory_manager.py`   | Temporary memory across UI actions |

---

## 🔁 Optional: Demo Script

Use the included `demo_script.md` to guide a video walkthrough for presentations or submissions.

---

## ⚙️ Dependencies

- Streamlit
- Transformers (Hugging Face)
- PyPDF2
- scikit-learn
- sentencepiece
- torch

---

## 🧠 Credits & Notes

Built to showcase contextual understanding and logical inference from unstructured documents. Ideal for research assistants, educational tools, and smart Q&A applications.


## 🎬 Project Demo

[![Watch on YouTube](https://img.shields.io/badge/Watch%20on-YouTube-red?style=for-the-badge&logo=youtube)](https://youtu.be/xwC8BKSzxtk)

🚀 **Check out the video demo** where I walk through the main features of the project, explain its purpose, and show it in action:

[![Project Demo Video](https://img.youtube.com/vi/xwC8BKSzxtk/maxresdefault.jpg)](https://youtu.be/xwC8BKSzxtk)

✨ Click the thumbnail or the button above to watch the full demo on YouTube!
