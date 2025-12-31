import os
from pathlib import Path

env_path = Path('rag_service/.env')
if env_path.exists():
    content = env_path.read_text()
    # Replace the placeholder or wrong model name
    # We'll look for LMSTUDIO_MODEL and update it
    new_lines = []
    for line in content.splitlines():
        if line.startswith('LMSTUDIO_MODEL='):
            new_lines.append('LMSTUDIO_MODEL=qwen/qwen2.5-vl-7b')
        else:
            new_lines.append(line)
    
    # If LMSTUDIO_MODEL wasn't found, add it
    if not any(l.startswith('LMSTUDIO_MODEL=') for l in new_lines):
        new_lines.append('LMSTUDIO_MODEL=qwen/qwen2.5-vl-7b')
        
    env_path.write_text('\n'.join(new_lines))
    print("Updated LMSTUDIO_MODEL in .env")
else:
    print(".env not found")
