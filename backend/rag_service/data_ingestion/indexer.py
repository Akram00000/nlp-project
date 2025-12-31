"""
Vector Indexer - Create embeddings and upload to Qdrant.

Handles the actual indexing of documents into vector database.
"""

from typing import Iterator, Union
from pathlib import Path
from loguru import logger
from tqdm import tqdm

from .fatwa_loader import FatwaDocument
from .hadith_loader import HadithDocument
from .book_loader import BookDocument


DocumentType = Union[FatwaDocument, HadithDocument, BookDocument]


class VectorIndexer:
    """
    Index documents into Qdrant vector database.
    
    Features:
    - Batch embedding generation
    - Efficient upload to Qdrant
    - Progress tracking
    - Separate collections for fatwas and hadiths
    """
    
    def __init__(
        self,
        qdrant_client=None,
        embedding_model=None,
        batch_size: int = 32,
    ):
        """
        Initialize indexer.
        
        Args:
            qdrant_client: Qdrant client instance
            embedding_model: Sentence transformer model
            batch_size: Batch size for embedding generation
        """
        self._qdrant = qdrant_client
        self._embedding_model = embedding_model
        self.batch_size = batch_size
        self._initialized = False
    
    def initialize(
        self,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        qdrant_api_key: str = None,
        embedding_model_name: str = "Omartificial-Intelligence-Space/GATE-AraBert-v1",
    ):
        """Initialize Qdrant and embedding model."""
        if self._qdrant is None:
            from qdrant_client import QdrantClient
            self._qdrant = QdrantClient(
                host=qdrant_host,
                port=qdrant_port,
                api_key=qdrant_api_key,
                https=False,  # Use HTTP for local Qdrant
                prefer_grpc=False,  # Use REST API instead of gRPC
            )
            logger.info(f"Connected to Qdrant at {qdrant_host}:{qdrant_port}")
        
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(embedding_model_name)
            logger.info(f"Loaded embedding model: {embedding_model_name}")
        
        self._initialized = True
    
    def create_collection(
        self,
        collection_name: str,
        vector_size: int = 768,
        force_recreate: bool = False,
    ):
        """Create Qdrant collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams
        
        if not self._initialized:
            self.initialize()
        
        # Check if collection exists
        collections = self._qdrant.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists and force_recreate:
            logger.warning(f"Deleting existing collection: {collection_name}")
            self._qdrant.delete_collection(collection_name)
            exists = False
        
        if not exists:
            self._qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )
            logger.info(f"Created collection: {collection_name}")
        else:
            logger.info(f"Collection already exists: {collection_name}")
    
    def index_documents(
        self,
        documents: Iterator[DocumentType] | list[DocumentType],
        collection_name: str,
        show_progress: bool = True,
    ) -> dict:
        """
        Index documents into Qdrant.
        
        Args:
            documents: Iterator or list of documents
            collection_name: Target collection name
            show_progress: Show progress bar
            
        Returns:
            Statistics about indexing operation
        """
        if not self._initialized:
            self.initialize()
        
        # Convert to list if iterator
        if not isinstance(documents, list):
            documents = list(documents)
        
        total = len(documents)
        logger.info(f"Indexing {total} documents into {collection_name}")
        
        if total == 0:
            return {'total': 0, 'indexed': 0}
        
        # Process in batches
        from qdrant_client.models import PointStruct
        
        indexed = 0
        failed = 0
        
        progress = tqdm(total=total, desc="Indexing") if show_progress else None
        
        for i in range(0, total, self.batch_size):
            batch = documents[i:i + self.batch_size]
            
            try:
                # Generate embeddings
                contents = [doc.content for doc in batch]
                embeddings = self._embedding_model.encode(
                    contents,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                )
                
                # Create points
                points = []
                for doc, embedding in zip(batch, embeddings):
                    point = PointStruct(
                        id=hash(doc.id) % (2**63),  # Convert to int ID
                        vector=embedding.tolist(),
                        payload={
                            'id': doc.id,
                            'content': doc.content,
                            **doc.metadata,
                        }
                    )
                    points.append(point)
                
                # Upload to Qdrant
                self._qdrant.upsert(
                    collection_name=collection_name,
                    points=points,
                )
                
                indexed += len(batch)
                
            except Exception as e:
                logger.error(f"Failed to index batch {i}: {e}")
                failed += len(batch)
            
            if progress:
                progress.update(len(batch))
        
        if progress:
            progress.close()
        
        logger.info(f"Indexing complete: {indexed} successful, {failed} failed")
        
        return {
            'total': total,
            'indexed': indexed,
            'failed': failed,
        }
    
    def index_fatwas(
        self,
        data_path: Path | str,
        collection_name: str = "fatwas",
        pattern: str = "*.jsonl",
        force_recreate: bool = False,
    ) -> dict:
        """
        Load and index fatwas from directory.
        
        Args:
            data_path: Path to fatwa JSON files
            collection_name: Collection name
            pattern: File pattern
            force_recreate: Recreate collection if exists
            
        Returns:
            Indexing statistics
        """
        from .fatwa_loader import FatwaLoader
        
        # Create collection
        self.create_collection(collection_name, force_recreate=force_recreate)
        
        # Load fatwas
        loader = FatwaLoader(
            max_chunk_size=0
        )
        documents = list(loader.load_directory(data_path, pattern))
        
        # Print statistics
        stats = loader.get_statistics(documents)
        logger.info(f"Loaded {stats['total_fatwas']} fatwas from {stats['unique_sources']} sources")
        
        # Index
        result = self.index_documents(documents, collection_name)
        result['fatwa_stats'] = stats
        
        return result
    
    def index_hadiths(
        self,
        data_path: Path | str,
        collection_name: str = "hadiths",
        pattern: str = "*.jsonl",
        force_recreate: bool = False,
    ) -> dict:
        """
        Load and index hadiths from directory.
        
        Args:
            data_path: Path to hadith JSON files
            collection_name: Collection name
            pattern: File pattern
            force_recreate: Recreate collection if exists
            
        Returns:
            Indexing statistics
        """
        from .hadith_loader import HadithLoader
        
        # Create collection
        self.create_collection(collection_name, force_recreate=force_recreate)
        
        # Load hadiths
        loader = HadithLoader()
        documents = list(loader.load_directory(data_path, pattern) if Path(data_path).is_dir() 
                        else loader.load_jsonl(data_path))
        
        # Print statistics
        stats = loader.get_statistics(documents)
        logger.info(f"Loaded {stats['total_hadiths']} hadiths from {stats['unique_sources']} sources")
        
        # Index
        result = self.index_documents(documents, collection_name)
        result['hadith_stats'] = stats
        
        return result

    def index_books(
        self,
        data_path: Path | str,
        collection_name: str = "books",
        madhab_filter: str | None = None,
        preprocess: bool = False,
        force_recreate: bool = False,
    ) -> dict:
        """
        Load and index books from directory.
        
        Args:
            data_path: Path to OCR book files (e.g., data/OCR_google_documentAI_...)
            collection_name: Collection name (default: "books")
            madhab_filter: Only index specific madhab (e.g., "hanafi", "maliki")
            preprocess: Apply OCR preprocessing
            force_recreate: Recreate collection if exists
            
        Returns:
            Indexing statistics
        """
        from .book_loader import BookLoader
        
        # Create collection
        self.create_collection(collection_name, force_recreate=force_recreate)
        
        # Create payload indices for book metadata
        self._create_book_indices(collection_name)
        
        # Load books
        loader = BookLoader(
            target_chunk_tokens=1000,
            chunk_overlap_tokens=100,
            preprocess=preprocess,
            extract_chapters=True,
        )
        documents = list(loader.load_directory(data_path, madhab_filter=madhab_filter))
        
        # Print statistics
        stats = loader.get_statistics(documents)
        logger.info(
            f"Loaded {stats['total_chunks']} chunks from {stats['unique_books']} books "
            f"({', '.join(stats['madhabs'])})"
        )
        
        # Index
        result = self.index_documents(documents, collection_name)
        result['book_stats'] = stats
        
        return result
    
    def _create_book_indices(self, collection_name: str):
        """Create payload indices for efficient book filtering."""
        try:
            from qdrant_client.models import PayloadSchemaType
            
            # Create indices for common filter fields
            index_fields = ['madhab', 'author', 'book_id', 'type', 'volume']
            
            for field in index_fields:
                try:
                    self._qdrant.create_payload_index(
                        collection_name=collection_name,
                        field_name=field,
                        field_schema=PayloadSchemaType.KEYWORD,
                    )
                    logger.debug(f"Created index for field: {field}")
                except Exception as e:
                    # Index might already exist
                    logger.debug(f"Index creation for {field}: {e}")
                    
        except Exception as e:
            logger.warning(f"Failed to create payload indices: {e}")
