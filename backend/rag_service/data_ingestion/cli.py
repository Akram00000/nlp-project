"""
CLI tool for indexing fatwas and hadiths.

Usage:
    # Index fatwas
    python -m backend.rag_service.data_ingestion.cli index-fatwas data/fatwas/

    # Index hadiths  
    python -m backend.rag_service.data_ingestion.cli index-hadiths data/hadiths/
    
    # Index both
    python -m backend.rag_service.data_ingestion.cli index-all data/
"""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table
from loguru import logger

from .indexer import VectorIndexer
from ..config.settings import settings


app = typer.Typer(help="Index fatwas and hadiths into vector database")
console = Console()


@app.command()
def index_fatwas(
    data_path: Path = typer.Argument(..., help="Path to fatwa JSON/JSONL files"),
    collection: str = typer.Option("fatwas", help="Collection name"),
    pattern: str = typer.Option("*.jsonl", help="File pattern"),
    force: bool = typer.Option(False, "--force", help="Recreate collection if exists"),
    host: str = typer.Option(None, help="Qdrant host (overrides config)"),
    port: int = typer.Option(None, help="Qdrant port (overrides config)"),
):
    """Index fatwas from JSON/JSONL files."""
    
    console.print(f"\n[bold blue]Indexing Fatwas[/bold blue]")
    console.print(f"Data path: {data_path}")
    console.print(f"Collection: {collection}\n")
    
    # Initialize indexer
    indexer = VectorIndexer()
    indexer.initialize(
        qdrant_host=host or settings.qdrant.host,
        qdrant_port=port or settings.qdrant.port,
        qdrant_api_key=settings.qdrant.api_key,
        embedding_model_name=settings.embedding.model_name,
    )
    
    # Index
    result = indexer.index_fatwas(
        data_path=data_path,
        collection_name=collection,
        pattern=pattern,
        force_recreate=force,
    )
    
    # Display results
    table = Table(title="Indexing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Fatwas", str(result['total']))
    table.add_row("Successfully Indexed", str(result['indexed']))
    table.add_row("Failed", str(result['failed']))
    
    if 'fatwa_stats' in result:
        stats = result['fatwa_stats']
        table.add_row("Unique Scholars", str(stats['unique_scholars']))
        table.add_row("Unique Sources", str(stats['unique_sources']))
        table.add_row("With References", f"{stats['with_references']} ({stats['reference_percentage']:.1f}%)")
    
    console.print(table)
    console.print("\n[bold green]✓ Fatwa indexing complete![/bold green]\n")


@app.command()
def index_hadiths(
    data_path: Path = typer.Argument(..., help="Path to hadith JSON/JSONL files"),
    collection: str = typer.Option("hadiths", help="Collection name"),
    pattern: str = typer.Option("*.json", help="File pattern"),
    force: bool = typer.Option(False, "--force", help="Recreate collection if exists"),
    host: str = typer.Option(None, help="Qdrant host"),
    port: int = typer.Option(None, help="Qdrant port"),
):
    """Index hadiths from JSON/JSONL files."""
    
    console.print(f"\n[bold blue]Indexing Hadiths[/bold blue]")
    console.print(f"Data path: {data_path}")
    console.print(f"Collection: {collection}\n")
    
    # Initialize indexer
    indexer = VectorIndexer()
    indexer.initialize(
        qdrant_host=host or settings.qdrant.host,
        qdrant_port=port or settings.qdrant.port,
        qdrant_api_key=settings.qdrant.api_key,
        embedding_model_name=settings.embedding.model_name,
    )
    
    # Index
    result = indexer.index_hadiths(
        data_path=data_path,
        collection_name=collection,
        pattern=pattern,
        force_recreate=force,
    )
    
    # Display results
    table = Table(title="Indexing Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Total Hadiths", str(result['total']))
    table.add_row("Successfully Indexed", str(result['indexed']))
    table.add_row("Failed", str(result['failed']))
    
    if 'hadith_stats' in result:
        stats = result['hadith_stats']
        table.add_row("Unique Sources", str(stats['unique_sources']))
        table.add_row("Unique Narrators", str(stats['unique_narrators']))
    
    console.print(table)
    console.print("\n[bold green]✓ Hadith indexing complete![/bold green]\n")


@app.command()
def index_all(
    data_dir: Path = typer.Argument(..., help="Base data directory"),
    fatwa_subdir: str = typer.Option("", help="Fatwa subdirectory (relative to data_dir)"),
    hadith_subdir: str = typer.Option("", help="Hadith subdirectory"),
    force: bool = typer.Option(False, "--force", help="Recreate collections"),
):
    """Index both fatwas and hadiths."""
    
    console.print("\n[bold blue]Indexing All Data[/bold blue]\n")
    
    # Index fatwas
    fatwa_path = data_dir / fatwa_subdir if fatwa_subdir else data_dir
    if fatwa_path.exists():
        console.print(f"[yellow]Step 1/2: Indexing fatwas from {fatwa_path}[/yellow]")
        indexer = VectorIndexer()
        indexer.initialize(
            qdrant_host=settings.qdrant.host,
            qdrant_port=settings.qdrant.port,
            qdrant_api_key=settings.qdrant.api_key,
        )
        indexer.index_fatwas(fatwa_path, force_recreate=force)
    
    # Index hadiths
    hadith_path = data_dir / hadith_subdir if hadith_subdir else data_dir / "hadiths"
    if hadith_path.exists():
        console.print(f"\n[yellow]Step 2/2: Indexing hadiths from {hadith_path}[/yellow]")
        indexer = VectorIndexer()
        indexer.initialize()
        indexer.index_hadiths(hadith_path, force_recreate=force)
    
    console.print("\n[bold green]✓ All indexing complete![/bold green]\n")


@app.command()
def check_references(
    fatwa_file: Path = typer.Argument(..., help="Fatwa file to analyze"),
    limit: int = typer.Option(10, help="Number of fatwas to check"),
):
    """Check what references are extracted from fatwas."""
    
    from .fatwa_loader import FatwaLoader
    from .reference_extractor import ReferenceExtractor
    
    console.print(f"\n[bold blue]Analyzing References in {fatwa_file}[/bold blue]\n")
    
    loader = FatwaLoader(extract_references=True)
    docs = list(loader.load_jsonl(fatwa_file))[:limit]
    
    extractor = ReferenceExtractor()
    
    for doc in docs:
        refs = extractor.extract_all(doc.content)
        
        if refs['hadiths'] or refs['ayahs']:
            console.print(f"\n[cyan]Fatwa ID: {doc.id}[/cyan]")
            
            if refs['hadiths']:
                console.print(f"  [yellow]Hadiths found: {len(refs['hadiths'])}[/yellow]")
                for h in refs['hadiths'][:3]:
                    console.print(f"    - {h['query'][:80]}...")
            
            if refs['ayahs']:
                console.print(f"  [yellow]Ayahs found: {len(refs['ayahs'])}[/yellow]")
                for a in refs['ayahs'][:3]:
                    console.print(f"    - {a['text'][:60]}... [{a.get('surah', '?')}]")


if __name__ == "__main__":
    app()
