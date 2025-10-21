from transformers import MarianMTModel, MarianTokenizer

# Dictionary of target languages and their Hugging Face models
models = {
    "French": "Helsinki-NLP/opus-mt-en-fr",
    "German": "Helsinki-NLP/opus-mt-en-de",
    "Italian": "Helsinki-NLP/opus-mt-en-it",
    "Dutch": "Helsinki-NLP/opus-mt-en-nl",
    "Hindi": "Helsinki-NLP/opus-mt-en-hi"
}

# Cache loaded models to avoid re-downloading
loaded_models = {}

def translate(text, target_lang):
    if target_lang not in loaded_models:
        model_name = models[target_lang]
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        loaded_models[target_lang] = (tokenizer, model)
    else:
        tokenizer, model = loaded_models[target_lang]

    tokens = tokenizer(text, return_tensors="pt", padding=True)
    translated = model.generate(**tokens)
    return tokenizer.decode(translated[0], skip_special_tokens=True)

# Main loop
print("=== Multi-Language Translator (English → Other Languages) ===")
print("Type 'quit' at any time to exit.\n")

while True:
    print("Available target languages:")
    for i, lang in enumerate(models.keys(), start=1):
        print(f"{i}. {lang}")
    
    lang_choice = input("Choose the target language (name or number): ")

    if lang_choice.lower() == "quit":
        print("Exiting translator. Goodbye!")
        break

    # Map number to language if user entered a number
    if lang_choice.isdigit():
        idx = int(lang_choice) - 1
        if idx < 0 or idx >= len(models):
            print("Invalid choice. Try again.\n")
            continue
        target_lang = list(models.keys())[idx]
    else:
        target_lang = lang_choice.title()
        if target_lang not in models:
            print("Invalid choice. Try again.\n")
            continue

    text = input(f"Enter text in English to translate to {target_lang}: ")
    if text.lower() == "quit":
        print("Exiting translator. Goodbye!")
        break

    translation = translate(text, target_lang)
    print(f"Translated text ({target_lang}): {translation}")
    print("-" * 50)

#Done! 