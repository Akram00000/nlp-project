"""
Restore Qdrant collections from snapshots.

Instructions:
1. Make sure Qdrant is running (Docker or standalone)
2. Place this script in the same directory as your qdrant_export folder
3. Update the Qdrant connection settings below if needed
4. Run: python restore_db.py
"""

from qdrant_client import QdrantClient
from pathlib import Path
import sys

# ============================================================
# CONFIGURATION - Update these if needed
# ============================================================
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_API_KEY = None  # Set if using Qdrant Cloud
EXPORT_DIR = Path("../qdrant_export")  # Path to your exported snapshots

# ============================================================
# RESTORE SCRIPT
# ============================================================

def main():
    print("="*60)
    print("Qdrant Database Restore")
    print("="*60)
    
    # Check if export directory exists
    if not EXPORT_DIR.exists():
        print(f"[X] Export directory not found: {EXPORT_DIR.absolute()}")
        print(f"\nMake sure the qdrant_export folder is in the correct location.")
        print(f"Current script location: {Path(__file__).parent.absolute()}")
        sys.exit(1)
    
    # Find all snapshots
    snapshots = []
    
    # Strategy: Look for *.snapshot files either directly in EXPORT_DIR/collection/ 
    # or in EXPORT_DIR/qdrant_export/collection/
    possible_roots = [EXPORT_DIR]
    nested_dir = EXPORT_DIR / "qdrant_export"
    if nested_dir.exists() and nested_dir.is_dir():
        possible_roots.append(nested_dir)
        
    for root in possible_roots:
        for collection_dir in root.iterdir():
            if collection_dir.is_dir():
                for snapshot_file in collection_dir.glob("*.snapshot"):
                    # Avoid duplicates if roots overlap (though they shouldn't here)
                    if not any(s['filename'] == snapshot_file.name for s in snapshots):
                        snapshots.append({
                            'collection': collection_dir.name,
                            'path': snapshot_file,
                            'filename': snapshot_file.name
                        })
    
    if not snapshots:
        print(f"[X] No snapshots found in {EXPORT_DIR.absolute()}")
        print(f"Searched in:")
        for r in possible_roots:
            print(f"  - {r.absolute()}")
        sys.exit(1)
    
    print(f"\nFound {len(snapshots)} snapshot(s) to restore:")
    for s in snapshots:
        print(f"  - {s['collection']}: {s['filename']}")
    
    # Connect to Qdrant
    print(f"\n{'='*60}")
    print(f"Connecting to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"{'='*60}")
    
    try:
        client = QdrantClient(
            host=QDRANT_HOST,
            port=QDRANT_PORT,
            api_key=QDRANT_API_KEY,
            timeout=300
        )
        # Test connection
        collections = client.get_collections()
        print(f"[V] Connected successfully")
        print(f"Existing collections: {[c.name for c in collections.collections]}")
    except Exception as e:
        print(f"[X] Failed to connect to Qdrant: {e}")
        print(f"\nMake sure Qdrant is running:")
        print(f"  Docker: docker run -p 6333:6333 qdrant/qdrant")
        print(f"  Or check your connection settings in this script")
        sys.exit(1)
    
    # Restore each snapshot
    print(f"\n{'='*60}")
    print(f"Restoring collections")
    print(f"{'='*60}")
    
    for snapshot in snapshots:
        collection_name = snapshot['collection']
        snapshot_path = snapshot['path']
        
        print(f"\nRestoring: {collection_name}")
        
        try:
            # Recover from snapshot by uploading the file
            print(f"  Uploading and restoring from {snapshot['filename']}...")
            with open(snapshot_path, "rb") as f:
                client.http.snapshots_api.recover_from_uploaded_snapshot(
                    collection_name=collection_name,
                    snapshot=f
                )
            
            # Verify restoration
            info = client.get_collection(collection_name)
            print(f"  [V] Restored successfully!")
            print(f"    - Vectors: {info.points_count}")
            print(f"    - Vector size: {info.config.params.vectors.size}")
            
        except Exception as e:
            print(f"  [X] Failed to restore {collection_name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"Restore complete!")
    print(f"{'='*60}")
    
    # Show final status
    collections = client.get_collections()
    print(f"\nFinal collections in database:")
    for collection in collections.collections:
        info = client.get_collection(collection.name)
        print(f"  - {collection.name}: {info.points_count} vectors")
    
    print(f"\n[V] All done! Your Qdrant database is ready to use.")


if __name__ == "__main__":
    main()
