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

movies = pd.read_parquet('movies_filtered.parquet')
corpus_texts = []

templates = [
    "{title} est un film {genres} sorti en {year}. Avec {actors}, il obtient une note de {rating}/10 sur IMDb.\n\n",
    "Film {genres} de {year}, {title} met en scène {actors}. Note IMDb: {rating}/10 ({votes} votes).\n\n",
    "{title} ({year}) - {genres}. Acteurs principaux: {actors}. Score: {rating}/10.\n\n",
    "En {year}, {title} a été réalisé avec {actors}. Ce film {genres} a une note de {rating} sur IMDb.\n\n",
    "Recommandation: {title} ({year}) - {genres}. Avec {actors}. IMDb: {rating}/10 ({votes} votes).\n\n",
]

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
    
    import random
    template = random.choice(templates)
    
    text = template.format(
        title=title,
        year=year,
        genres=genres,
        actors=actors,
        rating=rating,
        votes=f"{votes:,}"
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

    itype = random.choice(["reco", "resume", "acteurs"])

    if itype == "reco":
        instruction = f"Recommande-moi un bon film {genres.lower()}."
        output = (
            f"Tu peux regarder {title} ({year}). "
            f"C’est un film {genres.lower()} avec {actors}. "
            f"Il a une note de {rating}/10 sur IMDb avec {votes} votes."
        )

    elif itype == "resume":
        instruction = (
            f"Donne un court descriptif d’un film {genres.lower()} sorti en {year}."
        )
        output = (
            f"{title} est un film {genres.lower()} sorti en {year} avec {actors}. "
            f"Il est apprécié du public avec une note de {rating}/10."
        )

    else: 
        instruction = f"Quels sont les acteurs principaux du film {title} ?"
        output = f"Les principaux acteurs de {title} ({year}) sont {actors}."

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
