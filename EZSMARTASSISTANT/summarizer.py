from transformers import pipeline

# Load a pre-trained summarizer model
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

def generate_summary(text):
    if len(text) < 100:
        return "Text too short to summarize."

    # Limit input for model safety (1000 characters)
    input_text = text[:1000]

    try:
        summary = summarizer(input_text, max_length=150, min_length=140, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"Summarization error: {e}"
