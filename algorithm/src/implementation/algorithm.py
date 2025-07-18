from logging import getLogger
from pathlib import Path
from typing import Any, Optional, TypeVar
from oceanprotocol_job_details.ocean import JobDetails

# =============================== IMPORT LIBRARY ====================
import json
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
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

        df = pd.read_csv(filepath, encoding="ISO-8859-1", header=None)
        chunks = []

        for _, row in df.iterrows():
            if len(row) < 2:
                continue  # skip malformed
            raw_email = str(row[1]).strip()
            if raw_email:
                chunks.append(raw_email[:3000])  # prevent overlengths

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
        embeddings = np.array(embeddings).astype("float32")  # FAISS requires float32

        # === Step 5: Store vectors in ChromaDB ===
        # === Initialize ChromaDB ===
        logger.info("Indexing with FAISS...")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        faiss.write_index(index, "faiss_index.bin")
        logger.info("Saved FAISS index to faiss_index.bin")

        # === Save chunks to docstore.json ===
        with open("docstore.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, indent=2)
        logger.info("Chunk store saved as docstore.json")

        # TODO: 5. save results here
        # === Step 6: Save run result metadata ===
        self.results = {
            "status": "completed",
            "n_chunks": len(chunks),
            "docstore": "docstore.json",
            "faiss_index": "faiss_index.bin"
        }

        # TODO: 6. return self
        return self

    def save_result(self, path: Path) -> None:
        # TODO: 7. define/add result path here
        result_path = path / "result.json"


        with open(result_path, "w", encoding="utf-8") as f:
            try:
                # TODO: 8. save results here
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(self.results, f, indent=2)
                logger.info(f"Saved result metadata to {result_path}")
            except Exception as e:
                logger.exception(f"Error saving data: {e}")