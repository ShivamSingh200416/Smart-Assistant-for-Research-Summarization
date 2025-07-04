class MemoryManager:
    def __init__(self):
        self.context = ""
        self.temp_answer = ""

    def store_context(self, doc_text):
        self.context = doc_text

    def get_context(self):
        return self.context

    def store_temp_answer(self, answer):
        self.temp_answer = answer

    def clear_temp_answer(self):
        self.temp_answer = ""