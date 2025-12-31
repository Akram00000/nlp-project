# Qdrant Database Restoration Guide

This guide will help you restore the Qdrant database from the exported snapshots.

## Prerequisites

1. **Python 3.8+** installed
2. **Qdrant** running (Docker or standalone)
3. **qdrant-client** Python package

## Step 1: Install Dependencies

```bash
pip install qdrant-client
```

## Step 2: Start Qdrant

### Option A: Using Docker (Recommended)
```bash
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant
```

### Option B: Using Docker Compose
Create a `docker-compose.yml`:
```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_storage:/qdrant/storage
```

Then run:
```bash
docker-compose up -d
```

### Option C: Standalone Installation
Download and run Qdrant from: https://github.com/qdrant/qdrant/releases

## Step 3: Prepare Files

Your folder structure should look like this:
```
your_project/
├── restore_db.py
└── qdrant_export/
    ├── books/
    │   └── books-xxx.snapshot
    ├── fatwas/
    │   └── fatwas-xxx.snapshot
    └── hadiths/
        └── hadiths-xxx.snapshot
```

## Step 4: Configure Connection (if needed)

Edit `restore_db.py` if your Qdrant is not running on localhost:6333:

```python
QDRANT_HOST = "localhost"  # Change to your Qdrant host
QDRANT_PORT = 6333         # Change to your Qdrant port
QDRANT_API_KEY = None      # Set if using Qdrant Cloud
```

## Step 5: Run Restore Script

```bash
python restore_db.py
```

The script will:
1. Find all snapshots in the `qdrant_export` folder
2. Connect to your Qdrant instance
3. Ask for confirmation before overwriting existing collections
4. Restore each collection from its snapshot
5. Verify the restoration was successful

## Verification

After restoration, you can verify your collections:

### Using Python:
```python
from qdrant_client import QdrantClient

client = QdrantClient(host="localhost", port=6333)

# List all collections
collections = client.get_collections()
for collection in collections.collections:
    info = client.get_collection(collection.name)
    print(f"{collection.name}: {info.points_count} vectors")
```

### Using Qdrant Dashboard:
Open http://localhost:6333/dashboard in your browser

## Troubleshooting

### Error: "Failed to connect to Qdrant"
- Make sure Qdrant is running: `docker ps` (for Docker)
- Check if port 6333 is accessible: `curl http://localhost:6333/collections`
- Verify firewall settings

### Error: "No snapshots found"
- Check that `qdrant_export` folder is in the correct location
- Verify snapshot files exist and have `.snapshot` extension

### Error: "Failed to restore collection"
- Check Qdrant logs for detailed error messages
- Ensure you have enough disk space
- Try deleting the collection manually first

### Collections are empty after restore
- Check the snapshot file is not corrupted
- Verify the snapshot file size is reasonable (not 0 bytes)
- Check Qdrant version compatibility

## Notes

- The restore process will ask before overwriting existing collections
- Large collections may take several minutes to restore
- The original vector embeddings and metadata will be preserved
- Snapshots are compatible across Qdrant versions (within major versions)

## Support

If you encounter issues:
1. Check Qdrant logs: `docker logs <container_name>` (for Docker)
2. Verify snapshot file integrity
3. Ensure Qdrant version compatibility
4. Contact the database administrator who created the export

---

For more information, visit: https://qdrant.tech/documentation/
