# RecSys: Content-Based Book Recommender

This repository contains a starter implementation for building a content-based recommender system on the [Goodbooks-10k dataset](https://github.com/zygmuntz/goodbooks-10k). The goal is to recommend books based on their metadata (titles, authors, and tags) using TF–IDF features and cosine similarity.

## Dataset
Download the dataset locally (no data is committed to the repo):

1. From Kaggle: [goodbooks-10k](https://www.kaggle.com/datasets/zygmuntz/goodbooks-10k)
2. Extract the CSV files into a `data/` directory at the project root. The scripts look for any of these by default:
   - `data/books.csv` (also `data/goodbooks-10k/books.csv`, `goodbooks-10k/books.csv`, or `books.csv`)
   - `data/tags.csv` (also `data/goodbooks-10k/tags.csv`, `goodbooks-10k/tags.csv`, or `tags.csv`)
   - `data/book_tags.csv` (also `data/goodbooks-10k/book_tags.csv`, `goodbooks-10k/book_tags.csv`, or `book_tags.csv`)
   - `data/ratings.csv` (also `data/goodbooks-10k/ratings.csv`, `goodbooks-10k/ratings.csv`, or `ratings.csv`)

If your CSVs live somewhere else entirely, either pass the explicit paths via `--books-path`, `--tags-path`, and `--book-tags-path`,
or set an environment variable `GOODBOOKS_DATA_DIR` that points to the folder containing the files. The CLI will search that
directory before falling back to the repo root or current working directory.

The provided code works with the original column names from the dataset (e.g., `book_id`, `goodreads_book_id`, `original_title`, `authors`).

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt
```

> **Tip:** Use a clean virtual environment (not the base Conda/Anaconda Python) to avoid version conflicts such as
> `numpy` 2.x being mixed with `pandas`/`scikit-learn` wheels built for NumPy 1.x. If you previously installed packages
> system-wide or in another environment, recreate the venv and reinstall the pinned requirements above.

## Usage
You can generate recommendations either from a seed title, a user's historical ratings, or directly from user tag preferences.
If your Goodbooks CSVs live in `./data` (the default suggested layout), you can simply run the commands below without
adding extra flags. If your files are elsewhere, pass `--books-path`, `--tags-path`, and `--book-tags-path` to point to
them explicitly.

### Recommend from a seed title
Build the feature matrix (titles/authors + tag features) and ask for the single closest match:

```bash
python src/recommender.py \
  --books-path data/books.csv \
  --tags-path data/tags.csv \
  --book-tags-path data/book_tags.csv \
  --title "The Hobbit"
```

The script prints **one recommendation** (by default) for the requested title with a short note about why it was chosen. You can still adjust the number of results with `--top-k` if you want to see more, and search by partial title matches.

### Recommend from user tag preferences
Use a list of tag identifiers (optionally with weights) to describe a user's interests. The recommender constructs a TF–IDF tag profile for each book and scores cosine similarity to the provided user vector, then reports **only the top pick with an explanation** by default:

```bash
python src/recommender.py \
  --books-path data/books.csv \
  --tags-path data/tags.csv \
  --book-tags-path data/book_tags.csv \
  --preferred-tags "30574,11305,11557"
```

To weight specific tags more heavily, pass `--preferred-tag-weights` using `tag_id:weight` pairs:

```bash
python src/recommender.py \
  --books-path data/books.csv \
  --tags-path data/tags.csv \
  --book-tags-path data/book_tags.csv \
  --preferred-tag-weights "30574:1.0,11305:0.6,11557:0.8"
```

The output includes the strongest contributing tags per book (tag names are shown when `tags.csv` is provided).

### Interactive tag prompt (asks you for tags and shows matching history)
If you prefer to type tags at runtime, simply omit `--preferred-tags` and `--preferred-tag-weights`. The CLI will ask for tag
names or IDs, recommend **one book** for those tags (with an explanation), and optionally list books you've read that share the same tag signals:

```bash
python src/recommender.py \
  --books-path data/books.csv \
  --tags-path data/tags.csv \
  --book-tags-path data/book_tags.csv
```

At the prompts:
- Enter one or more tag names/IDs (comma-separated). Unknown tags are reported and skipped.
- After seeing recommendations, optionally enter titles you've read to see which of them align most with your tag choices.

This flow bases the recommendation on the provided tags and then surfaces similar-tag books the user already knows, making it
easy to sanity-check the model's choices.

### Recommend from a user's rating history
If you want the model to build the profile from books a user has already rated (from `ratings.csv`), provide a `user_id`.
The recommender will filter that user's ratings (>=4.0 by default), build a weighted tag profile from the books they liked,
and then suggest the single best-matching new book with similar tag signals.

```bash
python src/recommender.py \
  --books-path data/books.csv \
  --tags-path data/tags.csv \
  --book-tags-path data/book_tags.csv \
  --ratings-path data/ratings.csv \
  --user-id 123  # replace with a real user_id from ratings.csv
```

If you omit tag preferences and `--user-id`, the CLI will ask for a `user_id` interactively (when `ratings.csv` is
available) before falling back to the tag prompt. The output shows:

- **One recommendation** chosen from the overlap between the user's highly rated books and the tag space.
- **The rated books that influenced the decision**, listing both their `book_id` and `goodreads_book_id`.
- **A short explanation** highlighting the strongest tag signals shared between the recommended title and the user's
  history.

## Project Structure
- `src/data_loader.py` – Utilities for loading the Goodbooks CSV files and assembling the book–tag matrix.
- `src/feature_builder.py` – Functions to create TF–IDF features from titles/authors and normalize tag weights.
- `src/recommender.py` – CLI-ready content-based recommender built on cosine similarity.
- `requirements.txt` – Python dependencies for the starter implementation.

## Next Steps
- Enrich text features (e.g., add book descriptions or genres).
- Persist precomputed feature matrices for faster startup.
- Add evaluation using the ratings data for offline validation.
- Build a lightweight API or UI for interactive recommendations.
