import pandas as pd
import os

DATA_DIR = "./datas" 

def load_imdb_tsv(filename):
    filepath = os.path.join(DATA_DIR, filename)
    return pd.read_csv(filepath, sep='\t', dtype=str, na_values='\\N')

titles = load_imdb_tsv('title.basics.tsv.gz')
ratings = load_imdb_tsv('title.ratings.tsv.gz')
principals = load_imdb_tsv('title.principals.tsv.gz')
names = load_imdb_tsv('name.basics.tsv.gz')
akas = load_imdb_tsv('title.akas.tsv.gz')  

import pandas as pd

movies = titles[titles['titleType'] == 'movie'].copy()
movies = movies[(movies['startYear'] >= '1980') & (movies['startYear'] <= '2024')]
movies['startYear'] = movies['startYear'].astype(int)
movies['runtimeMinutes'] = pd.to_numeric(movies['runtimeMinutes'], errors='coerce')
movies = movies[movies['runtimeMinutes'] > 45]
movies = movies.merge(ratings, on='tconst', how='inner')
movies = movies[movies['numVotes'].astype(int) >= 5000]
movies['averageRating'] = pd.to_numeric(movies['averageRating'], errors='coerce')
movies['numVotes'] = pd.to_numeric(movies['numVotes'], errors='coerce')

principals['ordering'] = pd.to_numeric(principals['ordering'], errors='coerce')
actors_in_films = principals[
    (principals['category'].isin(['actor', 'actress'])) &
    (principals['ordering'] <= 5)  
].copy()

actors_in_films = actors_in_films.merge(
    names[['nconst', 'primaryName']], 
    on='nconst', 
    how='left'
)

actors_by_film = actors_in_films.groupby('tconst')['primaryName'].apply(
    lambda x: ', '.join(x.dropna().unique())
).reset_index()
actors_by_film.columns = ['tconst', 'actors']

movies = movies.merge(actors_by_film, on='tconst', how='left')
movies['actors'] = movies['actors'].fillna('Unknown')
movies.to_parquet('movies_filtered.parquet', index=False)
movies.to_csv('movies_filtered.csv', index=False)

import pandas as pd
from tqdm import tqdm
import random

movies = pd.read_parquet('movies_filtered.parquet')
corpus_texts = []

templates = [
    # --- without scores ---
    "{title} est un film {genres} sorti en {year}, porté par {actors}. Il mise surtout sur l’ambiance et le récit pour immerger le spectateur.\n\n",
    "Sorti en {year}, {title} est un long-métrage {genres} avec {actors}. Le film se distingue par son ton particulier et sa mise en scène.\n\n",
    "{title} ({year}) est un film {genres}. Avec {actors}, il propose une histoire qui marie émotions et divertissement.\n\n",
    "Dans {title}, sorti en {year}, {actors} incarnent des personnages marquants. Ce film {genres} s’adresse surtout à ceux qui aiment les univers travaillés.\n\n",
    "{title} est une production {genres} de {year}. Le casting, mené par {actors}, donne une identité forte au film.\n\n",
    "{title}, sorti en {year}, appartient au genre {genres}. Il s’appuie sur {actors} pour donner vie à son récit.\n\n",
    "Avec {actors} au casting, {title} propose une expérience {genres} sortie en {year}, centrée sur ses personnages et son atmosphère.\n\n",

    # --- quality scores ---
    "{title} est un film {genres} sorti en {year} avec {actors}. Il est généralement {appreciation}.\n\n",
    "Avec {actors} en tête d’affiche, {title} propose un récit {genres} sorti en {year}. Le film est considéré comme {appreciation}.\n\n",
    "{title} ({year}) appartient au genre {genres}. Grâce à {actors}, le film a laissé une impression {appreciation} sur son public.\n\n",
    "Parmi les films {genres} sortis en {year}, {title} se distingue par son casting ({actors}) et un accueil {appreciation}.\n\n",
    "{title} est souvent cité comme un exemple {genres} {appreciation}, notamment grâce à la performance de {actors}.\n\n",

    # --- with score ---
    "{title} est un film {genres} sorti en {year} avec {actors}. Sur IMDb, il bénéficie d’une note d’environ {rating}/10, signe d’un intérêt réel du public.\n\n",
    "Film {genres} de {year}, {title} réunit {actors}. Sa note autour de {rating}/10 sur IMDb reflète des retours globalement positifs.\n\n",
    "{title} ({year}) met en avant {actors} dans un récit {genres}. La communauté IMDb lui attribue une note proche de {rating}/10.\n\n",
    "{title} est considéré comme un film {genres} solide. Sorti en {year} avec {actors}, il est évalué à environ {rating}/10 par les utilisateurs d’IMDb.\n\n",
    "Parmi les films {genres}, {title} ({year}) avec {actors} obtient une note avoisinant {rating}/10 sur IMDb, ce qui traduit son accueil.\n\n",
    "{title}, film {genres} sorti en {year}, met en scène {actors}. Sa note sur IMDb tourne autour de {rating}/10, ce qui reste cohérent avec les avis du public.\n\n",

    # --- Recommendation ---
    "Si tu cherches un film {genres} sorti en {year}, {title} avec {actors} est une option intéressante à considérer.\n\n",
    "Pour une soirée {genres}, {title} ({year}) avec {actors} peut être un bon choix, souvent {appreciation} par ceux qui l’ont vu.\n\n",
]

def rating_to_appreciation(rating):
    try:
        r = float(rating)
    except (TypeError, ValueError):
        return "accueilli de manière mitigée"
    if r >= 8.0:
        return "très apprécié du public"
    elif r >= 7.0:
        return "bien accueilli par les spectateurs"
    elif r >= 6.0:
        return "reçu de façon mitigée mais intéressant pour certains"
    else:
        return "plutôt mal reçu par le public"

def format_genres(genres_str):
    if pd.isna(genres_str) or genres_str == 'Unknown':
        return 'inconnu'
    genres = genres_str.split(',')
    if len(genres) == 1:
        return f"de {genres[0].lower()}"
    elif len(genres) == 2:
        return f"de {genres[0].lower()} et de {genres[1].lower()}"
    else:
        return f"de {', '.join(g.lower() for g in genres[:-1])} et de {genres[-1].lower()}"

def format_actors(actors_str, max_actors=3):
    if pd.isna(actors_str) or actors_str == 'Unknown':
        return "des acteurs inconnus"
    actors = actors_str.split(', ')
    if len(actors) > max_actors:
        actors = actors[:max_actors]
    if len(actors) == 1:
        return actors[0]
    elif len(actors) == 2:
        return f"{actors[0]} et {actors[1]}"
    else:
        return f"{', '.join(actors[:-1])} et {actors[-1]}"

for idx, row in tqdm(movies.iterrows(), total=len(movies)):
    title = row['primaryTitle']
    year = row['startYear']
    genres = format_genres(row['genres'])
    actors = format_actors(row['actors'])
    rating = row['averageRating']
    votes = row['numVotes']
    appreciation = rating_to_appreciation(rating)

    template = random.choice(templates)

    text = template.format(
        title=title,
        year=year,
        genres=genres,
        actors=actors,
        rating=rating,
        votes=f"{votes:,}",
        appreciation=appreciation,
    )
    corpus_texts.append(text)


print(f"{len(corpus_texts)} texts")

import pickle
with open('corpus_texts.pkl', 'wb') as f:
    pickle.dump(corpus_texts, f)

with open('corpus_sample.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(corpus_texts[:100]))  

import re
from datasets import Dataset

with open('corpus_texts.pkl', 'rb') as f:
    corpus_texts = pickle.load(f)

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\xa0', ' ')
    return text.strip()

cleaned_texts = [clean_text(t) for t in corpus_texts]

dataset = Dataset.from_dict({"text": cleaned_texts})

print(f"dataset size : {len(dataset)}")

dataset.save_to_disk('imdb_corpus_dataset')
print("saved")

import pandas as pd
import random
import json

movies = pd.read_parquet("movies_filtered.parquet")

instructions = []

def pick_random_movie(df):
    return df.sample(1).iloc[0]

for _ in range(5000):
    row = pick_random_movie(movies)
    title = row["primaryTitle"]
    year = row["startYear"]
    genres = row["genres"]
    actors = row["actors"]
    rating = row["averageRating"]
    votes = row["numVotes"]
    appreciation = rating_to_appreciation(rating)

    itype = random.choice(["reco", "resume", "actors", "popularity", "vibe"])

    if itype == "reco":
        instruction = random.choice([
            f"Recommande-moi un bon film {genres.lower()} à regarder ce soir.",
            f"Je cherche un film {genres.lower()} récent à voir. Tu me proposes quoi ?",
            f"Si j’aime les films {genres.lower()}, quel film me conseilles-tu ?",
        ])
        output = (
            f"Tu peux regarder {title} ({year}). "
            f"C’est un film {genres.lower()} avec {actors}. "
            f"Il est {appreciation}."
        )

    elif itype == "resume":
        instruction = random.choice([
            f"Donne un court descriptif d’un film {genres.lower()} sorti en {year}.",
            f"Résume brièvement un film {genres.lower()} avec {actors}.",
            f"Présente en quelques phrases un film {genres.lower()} marquant des années {year}.",
        ])
        output = (
            f"{title} est un film {genres.lower()} sorti en {year} avec {actors}. "
            f"Il raconte une histoire typique de ce genre et est {appreciation}."
        )

    elif itype == "actors":
        instruction = random.choice([
            f"Quels sont les acteurs principaux du film {title} ?",
            f"Qui joue dans le film {title} ({year}) ?",
            f"Donne-moi les acteurs principaux de {title}.",
        ])
        output = f"Les principaux acteurs de {title} ({year}) sont {actors}."

    elif itype == "popularity":
        instruction = random.choice([
            f"Explique pourquoi un film comme {title} est connu du grand public.",
            f"Pourquoi {title} est-il autant cité parmi les films {genres.lower()} ?",
            f"Qu’est-ce qui peut expliquer le succès de {title} ({year}) ?",
        ])
        output = (
            f"{title} ({year}) est un film {genres.lower()} avec {actors}. "
            f"Il est {appreciation}, ce qui explique qu’il soit souvent recommandé."
        )

    else:  # vibe
        instruction = random.choice([
            f"Décris l’ambiance générale d’un film {genres.lower()} comme {title}.",
            f"Quel type d’ambiance peut-on attendre d’un film {genres.lower()} tel que {title} ?",
            f"Parle-moi de l’atmosphère d’un film {genres.lower()} sorti en {year}.",
        ])
        output = (
            f"{title} est un film {genres.lower()} sorti en {year} avec {actors}. "
            f"L’ambiance correspond bien à ce qu’on attend d’un film de ce genre."
        )


    instructions.append(
        {
            "instruction": instruction,
            "input": "",
            "output": output,
        }
    )

with open("imdb_instructions.jsonl", "w", encoding="utf-8") as f:
    for ex in instructions:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Dataset d’instructions créé: {len(instructions)} exemples")

from datasets import load_dataset

dataset = load_dataset("json", data_files="imdb_instructions.jsonl")["train"]

def format_example(example):
    return (
        f"Instruction: {example['instruction']}\n"
        f"Réponse: {example['output']}\n\n"
    )

formatted_texts = [format_example(ex) for ex in dataset]

from datasets import Dataset
hf_dataset = Dataset.from_dict({"text": formatted_texts})
hf_dataset.save_to_disk("imdb_instructions_dataset")
print(hf_dataset[0]["text"])
