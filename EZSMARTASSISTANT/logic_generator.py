import random

def generate_logic_question(text):
    templates = [
        ("If all A are B, and all B are C, what can we say about A and C?", "All A are C."),
        ("If some D are E, and no E are F, can some D be F?", "No, because E and F are disjoint."),
        ("If X implies Y, and Y implies Z, what does X imply?", "X implies Z."),
        ("If not all M are N, and all N are O, what can we say about M and O?", "Some M may not be O."),
        ("If P and Q are true, and P is false, what can be said about Q?", "Nothing definitive about Q.")
    ]
    return random.choice(templates)

def evaluate_answer(user_input, correct_answer):
    user_input = user_input.strip().lower()
    correct = correct_answer.strip().lower()

    if correct in user_input or user_input in correct:
        return "✅ Correct! You understood the logical implication well."
    elif any(word in user_input for word in ["not", "no", "never", "cannot"]) and \
         any(word in correct for word in ["not", "no", "never", "cannot"]):
        return "🟨 Partially correct—your reasoning may align, but recheck the phrasing."
    else:
        return f"❌ Not quite. The ideal answer was: _{correct_answer}_"
