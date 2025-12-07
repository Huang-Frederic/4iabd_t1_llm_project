from transformers import pipeline
import torch


model_path = "./distilgpt2-imdb-finetuned"

generator = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    torch_dtype=torch.float16,
    device=0 if torch.cuda.is_available() else -1  
)

def generate_response(prompt, max_length=150, num_sequences=3):
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")
    
    outputs = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    
    for i, out in enumerate(outputs, 1):
        generated_text = out["generated_text"][len(prompt):].strip()
        print(f"\n--- Génération {i} ---")
        print(generated_text)
    
    return outputs

generate_response(
    "Propose-moi un film d'action récent avec de bons effets spéciaux.",
    max_length=150
)

generate_response(
    "Je veux un film avec Tom Cruise. Que me conseilles-tu ?",
    max_length=150
)

generate_response(
    "Donne-moi un court synopsis d'un film de science-fiction sorti dans les années 2000.",
    max_length=200
)

generate_response(
    "Quelle est la note du film Inception et pourquoi est-il si populaire ?",
    max_length=150
)

generate_response(
    "J'ai envie d'un film drôle et intelligent. Une idée ?",
    max_length=150
)

generate_response(
    "Liste les acteurs principaux du film Titanic.",
    max_length=150
)

print("\n\n" + "========== Instrictions finetuned ==========" + "\n\n")

model_path = "./distilgpt2-imdb-instructions-finetuned"

generator = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    torch_dtype=torch.float16,
    device=0 if torch.cuda.is_available() else -1  
)

def generate_response(prompt, max_length=150, num_sequences=3):
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")
    
    outputs = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    
    for i, out in enumerate(outputs, 1):
        generated_text = out["generated_text"][len(prompt):].strip()
        print(f"\n--- Génération {i} ---")
        print(generated_text)
    
    return outputs

generate_response(
    "Propose-moi un film d'action récent avec de bons effets spéciaux.",
    max_length=150
)

generate_response(
    "Je veux un film avec Tom Cruise. Que me conseilles-tu ?",
    max_length=150
)

generate_response(
    "Donne-moi un court synopsis d'un film de science-fiction sorti dans les années 2000.",
    max_length=200
)

generate_response(
    "Quelle est la note du film Inception et pourquoi est-il si populaire ?",
    max_length=150
)

generate_response(
    "J'ai envie d'un film drôle et intelligent. Une idée ?",
    max_length=150
)

generate_response(
    "Liste les acteurs principaux du film Titanic.",
    max_length=150
)

print("\n\n" + "========== gpt2-imdb-lora ==========" + "\n\n")

model_path = "./gpt2-imdb-lora"

generator = pipeline(
    "text-generation",
    model=model_path,
    tokenizer=model_path,
    torch_dtype=torch.float16,
    device=0 if torch.cuda.is_available() else -1  
)

def generate_response(prompt, max_length=150, num_sequences=3):
    print(f"\n{'='*60}")
    print(f"PROMPT: {prompt}")
    print(f"{'='*60}")
    
    outputs = generator(
        prompt,
        max_length=max_length,
        num_return_sequences=num_sequences,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.2,
        pad_token_id=generator.tokenizer.eos_token_id,
    )
    
    for i, out in enumerate(outputs, 1):
        generated_text = out["generated_text"][len(prompt):].strip()
        print(f"\n--- Génération {i} ---")
        print(generated_text)
    
    return outputs

generate_response(
    "Propose-moi un film d'action récent avec de bons effets spéciaux.",
    max_length=150
)

generate_response(
    "Je veux un film avec Tom Cruise. Que me conseilles-tu ?",
    max_length=150
)

generate_response(
    "Donne-moi un court synopsis d'un film de science-fiction sorti dans les années 2000.",
    max_length=200
)

generate_response(
    "Quelle est la note du film Inception et pourquoi est-il si populaire ?",
    max_length=150
)

generate_response(
    "J'ai envie d'un film drôle et intelligent. Une idée ?",
    max_length=150
)

generate_response(
    "Liste les acteurs principaux du film Titanic.",
    max_length=150
)
