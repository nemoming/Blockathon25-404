from logging import getLogger
from pathlib import Path
from typing import Any, Optional, TypeVar
from oceanprotocol_job_details.ocean import JobDetails

# =============================== IMPORT LIBRARY ====================
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
# =============================== END ===============================

T = TypeVar("T")

logger = getLogger(__name__)


class Algorithm:
    # TODO: [optional] add class variables here

    def __init__(self, job_details: JobDetails):
        self._job_details = job_details
        self.results = None

    def _validate_input(self) -> None:
        if not self._job_details.files or not self._job_details.files.files[0].input_files:
            logger.warning("No files found")
            raise ValueError("No files found")

    def _process_gazette(self, filepath: str) -> list[str]:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = []

        for decree in data.get("Decrees", []):
            title = decree.get("Title_EN", "")
            signer = decree.get("Signed_By", "")
            date = decree.get("Date", "")
            header = f"[Decree] {title} — Signed by {signer} on {date}"
            chunks.append(header)

            for article in decree.get("Articles", []):
                article_no = article.get("Article_No", "")
                content = article.get("Content_EN", "")
                text = f"Article {article_no}: {content}"
                chunks.append(text)

        for order in data.get("Orders", []):
            title = order.get("Title_EN", "")
            signer = order.get("Signed_By", "")
            date = order.get("Date", "")
            header = f"[Order] {title} — Signed by {signer} on {date}"
            chunks.append(header)

            for article in order.get("Articles", []):
                article_no = article.get("Article_No", "")
                content = article.get("Content_EN", "")
                text = f"Article {article_no}: {content}"
                chunks.append(text)

        return chunks
    
    def _process_enron(self, filepath: str) -> list[str]:
        df = pd.read_csv(filepath)

        # Normalize all column names to lowercase once
        df.columns = [col.lower() for col in df.columns]

        chunks = []

        for _, row in df.iterrows():
            sender = row.get("x-from", "N/A")
            recipient = row.get("x-to", "N/A")
            subject = row.get("x-subject", "No Subject")
            body = row.get("x-body", row.get("content", "No Body"))

            email = (
                f"From: {sender}\n"
                f"To: {recipient}\n"
                f"Subject: {subject}\n"
                f"Body:\n{body}"
            )

            chunks.append(email)

        return chunks

   
    def run(self) -> "Algorithm":
        # TODO: 1. Initialize results type 
        # === Step 1: Init and validate ===
        # self.results = {}
        self.results = {"status": "initialized"}

        # TODO: 2. validate input here
        self._validate_input()

        # TODO: 3. get input files here
        # === Step 2: Get the input file ===
        input_files = self._job_details.files.files[0].input_files
        filename = str(input_files[0])
        source = "gazette" if filename.endswith(".json") else "enron"
        logger.info(f"Input file detected: {filename}")

        # === Step 3: Identify dataset and generate text chunks ===
        if filename.endswith(".json"):
            logger.info("Detected Gazette JSON format")
            chunks = self._process_gazette(filename)
        elif filename.endswith(".csv"):
            logger.info("Detected Enron CSV format")
            chunks = self._process_enron(filename)
        else:
            raise ValueError("Unsupported file format: only JSON or CSV are supported")
    
        # TODO: 4. run algorithm here
        # === Step 4: Embed the text chunks ===
        logger.info("Loading embedding model...")
        model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Generating embeddings...")
        embeddings = model.encode(chunks, convert_to_numpy=True)

        # === Step 5: Store vectors in ChromaDB ===
        # === Initialize ChromaDB ===
        client = chromadb.PersistentClient(path="chroma_store")
        collection = client.get_or_create_collection("archive_chunks")

        logger.info("Inserting into Chroma collection...")
        metadatas = [{"index": i, "source": source} for i in range(len(chunks))]
        collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            metadatas=metadatas
)

        #client.persist()
        logger.info("ChromaDB persisted to disk at ./chroma_store")

        # === Save chunks to docstore.json ===
        with open("docstore.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        logger.info("Chunk store saved as docstore.json")

        # TODO: 5. save results here
        # === Step 7: Save run result metadata ===
        self.results = {
            "status": "completed",
            "n_chunks": len(chunks),
            "docstore": "docstore.json",
            "chroma_store": "chroma_store/"
        }

        # TODO: 6. return self
        return self

    def save_result(self, path: Path) -> None:
        # TODO: 7. define/add result path here
        result_path = path / "result.json"


        with open(result_path, "w", encoding="utf-8") as f:
            try:
                # TODO: 8. save results here
                json.dump(self.results, f, indent=2)
                logger.info(f"Saved result metadata to {result_path}")
                pass
            except Exception as e:
                logger.exception(f"Error saving data: {e}")
