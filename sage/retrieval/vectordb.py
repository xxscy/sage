"""
Everything related to vectordbs
"""
import contextlib
import gc
import os
import time
from pathlib import Path
from typing import Dict, List

import shutil
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma, FAISS

from sage.utils.common import CONSOLE


def _safe_remove_dir(vector_dir: str, retries: int = 5, wait_seconds: float = 1.0) -> None:
    """Attempt to remove a directory while dealing with Windows file locking."""

    if not os.path.isdir(vector_dir):
        return

    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(vector_dir)
            CONSOLE.log(f"Removed existing vector db at {vector_dir}")
            return
        except PermissionError as exc:
            CONSOLE.log(
                f"[yellow]Warning: attempt {attempt}/{retries} to delete {vector_dir} failed: {exc}"
            )
            gc.collect()
            time.sleep(wait_seconds)

    # fallback: delete files individually
    CONSOLE.log(f"[yellow]Falling back to manual cleanup for {vector_dir}")
    for root, dirs, files in os.walk(vector_dir, topdown=False):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                os.unlink(file_path)
            except PermissionError:
                backup = f"{file_path}.bak"
                try:
                    os.rename(file_path, backup)
                    os.unlink(backup)
                except Exception as exc:
                    CONSOLE.log(f"[red]Error removing locked file {file_path}: {exc}")
        for dirname in dirs:
            dir_path = os.path.join(root, dirname)
            with contextlib.suppress(OSError):
                os.rmdir(dir_path)
    with contextlib.suppress(OSError):
        os.rmdir(vector_dir)


def build_chroma_db(
    vector_dir: str,
    documents: List[Document],
    embeddings: Embeddings,
    load: bool = True,
) -> Chroma:
    """Create or load a chroma database with safer cleanup on Windows."""

    os.makedirs(vector_dir, exist_ok=True)

    if not load and os.path.isdir(vector_dir):
        _safe_remove_dir(vector_dir)
        os.makedirs(vector_dir, exist_ok=True)

    for attempt in range(5):
        try:
            if load and os.path.isdir(vector_dir) and os.listdir(vector_dir):
                CONSOLE.log(f"Loading vector db from {vector_dir}....")
                db = Chroma(
                    persist_directory=vector_dir,
                    embedding_function=embeddings,
                )

                if documents:
                    db.add_documents(documents)
                    if hasattr(db, "persist"):
                        db.persist()
                return db

            CONSOLE.log(f"Creating vector db in {vector_dir}...")
            db = Chroma.from_documents(
                documents or [Document(page_content="")],
                embeddings,
                persist_directory=vector_dir,
            )
            if hasattr(db, "persist"):
                db.persist()
            return db
        except PermissionError as exc:
            CONSOLE.log(f"[yellow]Warning: attempt {attempt+1}/5 failed: {exc}")
            gc.collect()
            time.sleep(1)

    CONSOLE.log(f"[red]Unable to create or load Chroma db at {vector_dir}, using fallback")
    fallback_dir = f"{vector_dir}_fallback"
    os.makedirs(fallback_dir, exist_ok=True)
    return Chroma.from_documents(
        [Document(page_content="")],
        embeddings,
        persist_directory=fallback_dir,
    )


def build_faiss_db(
    vector_dir,
    documents: List[Document],
    embeddings: Embeddings,
    load: bool = True,
):
    """Creates or loads a FAISS index"""

    if os.path.isdir(vector_dir) and load:
        CONSOLE.log(f"Loading vector db from {vector_dir}....")
        index = FAISS.load_local("smartie-index", embeddings)

        return index

    if not load:
        files = Path(vector_dir).glob("sage-index.*")

        if files:
            for filename in files:
                filename.unlink()

    CONSOLE.log("Creating vector db ...")
    index = FAISS.from_documents(documents=documents, embedding=embeddings)
    index.save_local(folder_path=vector_dir, index_name="sage-index")

    return index


VECTORDBS = {"chroma": build_chroma_db, "faiss": build_faiss_db}


def create_multiuser_vector_indexes(
    vectordb: str,
    documents: Dict[str, List[Document]],
    embedding_model: Embeddings,
    load: bool = True,
):
    """Creates a vector index that offers similarity search"""

    user_indexes = {}

    smarthome_root = os.getenv("SMARTHOME_ROOT")
    if not smarthome_root:
        CONSOLE.log("[red]SMARTHOME_ROOT environment variable is not set")
        return user_indexes

    for user_name, memories in documents.items():
        user_index_dir = os.path.join(smarthome_root, "user_info", user_name, vectordb)

        try:
            user_indexes[user_name] = VECTORDBS[vectordb](
                user_index_dir, memories, embedding_model, load=load
            )
        except Exception as exc:
            CONSOLE.log(
                f"[red]Failed to create vector index for user {user_name}: {exc}"
            )

    return user_indexes
