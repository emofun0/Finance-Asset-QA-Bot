from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class StoredIndex:
    vectorizer: TfidfVectorizer
    matrix: sparse.spmatrix


class LocalVectorStore:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.vectorizer_path = index_dir / "vectorizer.joblib"
        self.matrix_path = index_dir / "matrix.npz"

    def build(self, texts: list[str]) -> StoredIndex:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            min_df=1,
            sublinear_tf=True,
        )
        matrix = vectorizer.fit_transform(texts)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(vectorizer, self.vectorizer_path)
        sparse.save_npz(self.matrix_path, matrix)
        return StoredIndex(vectorizer=vectorizer, matrix=matrix)

    def load(self) -> StoredIndex:
        vectorizer: TfidfVectorizer = joblib.load(self.vectorizer_path)
        matrix = sparse.load_npz(self.matrix_path)
        return StoredIndex(vectorizer=vectorizer, matrix=matrix)
