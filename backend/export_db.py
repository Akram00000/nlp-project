"""
Export all Qdrant collections as snapshots (Docker version).
"""
from qdrant_client import QdrantClient
from pathlib import Path
import subprocess
import json
import time

# Connect to Qdrant with longer timeout
client = QdrantClient(host="localhost", port=6333, timeout=300)

# Get all collections
collections = client.get_collections()
print(f"Found {len(collections.collections)} collections:")
for c in collections.collections:
    print(f"  - {c.name}")

snapshots = []
for collection in collections.collections:
    collection_name = collection.name
    print(f"\nCreating snapshot for: {collection_name}")
    
    max_retries = 2
    for attempt in range(max_retries):
        try:
            # Create snapshot
            snapshot_info = client.create_snapshot(collection_name=collection_name)
            snapshots.append({
                'collection': collection_name,
                'snapshot': snapshot_info.name
            })
            print(f"✓ Snapshot created: {snapshot_info.name}")
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠ Attempt {attempt + 1} failed, retrying... ({e})")
                time.sleep(2)
            else:
                print(f"✗ Error creating snapshot for {collection_name}: {e}")

# Find Qdrant container
print(f"\n{'='*60}")
print("Finding Qdrant Docker container...")
print(f"{'='*60}")

try:
    result = subprocess.run(
        ["docker", "ps", "--filter", "ancestor=qdrant/qdrant", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        check=True
    )
    container_names = result.stdout.strip().split('\n')
    
    if not container_names or not container_names[0]:
        # Try finding by port
        result = subprocess.run(
            ["docker", "ps", "--filter", "publish=6333", "--format", "{{.Names}}"],
            capture_output=True,
            text=True,
            check=True
        )
        container_names = result.stdout.strip().split('\n')
    
    if not container_names or not container_names[0]:
        print("✗ No Qdrant container found!")
        print("\nManual steps:")
        print("1. Find your container: docker ps")
        print("2. Copy snapshots: docker cp <container>:/qdrant/storage/snapshots ./qdrant_export")
        exit(1)
    
    container_name = container_names[0]
    print(f"✓ Found container: {container_name}")
    
    # Check actual snapshot location in container
    print(f"\nChecking snapshot storage location in container...")
    result = subprocess.run(
        ["docker", "exec", container_name, "ls", "-la", "/qdrant/storage/snapshots/"],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"✓ Snapshots directory exists")
        print(result.stdout)
    else:
        print(f"⚠ Could not list snapshots directory")
    
except subprocess.CalledProcessError as e:
    print(f"✗ Error running docker command: {e}")
    print("Make sure Docker is running and accessible from PowerShell")
    exit(1)

# Create export directory
export_dir = Path("../qdrant_export")
export_dir.mkdir(exist_ok=True)

print(f"\n{'='*60}")
print(f"Copying snapshots from container to: {export_dir.absolute()}")
print(f"{'='*60}")

# Copy each snapshot from container
for snapshot_info in snapshots:
    collection_name = snapshot_info['collection']
    snapshot_name = snapshot_info['snapshot']
    
    # Create collection subdirectory
    collection_dir = export_dir / collection_name
    collection_dir.mkdir(exist_ok=True)
    
    # First, list files to find the actual snapshot location
    list_cmd = ["docker", "exec", container_name, "find", "/qdrant", "-name", snapshot_name]
    try:
        result = subprocess.run(list_cmd, capture_output=True, text=True, check=True)
        actual_paths = [p for p in result.stdout.strip().split('\n') if p]
        
        if not actual_paths:
            print(f"✗ Snapshot not found in container: {snapshot_name}")
            continue
        
        container_path = f"{container_name}:{actual_paths[0]}"
        print(f"Found: {actual_paths[0]}")
    except subprocess.CalledProcessError:
        # Try default path
        container_path = f"{container_name}:/qdrant/storage/snapshots/{collection_name}/{snapshot_name}"
    
    dest_path = str(collection_dir / snapshot_name)
    
    try:
        subprocess.run(
            ["docker", "cp", container_path, dest_path],
            check=True,
            capture_output=True
        )
        print(f"✓ Copied {collection_name}/{snapshot_name}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to copy {collection_name}/{snapshot_name}")
        print(f"  Error: {e.stderr.decode() if e.stderr else str(e)}")

print(f"\n{'='*60}")
print(f"Export complete!")
print(f"{'='*60}")
print(f"\nAll snapshots are in: {export_dir.absolute()}")
print(f"\nTo share with someone:")
print(f"  - Zip the folder: Compress-Archive -Path '{export_dir.absolute()}' -DestinationPath qdrant_backup.zip")
print(f"\nTo restore on another machine:")
print(f"  1. Start Qdrant (Docker or standalone)")
print(f"  2. For each collection, run:")
print(f"     client.recover_snapshot(collection_name='<name>', snapshot_path='<path>')")
print(f"     or use the Qdrant API/UI to upload snapshots")