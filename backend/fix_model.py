"""Fix the model name in .env to include the full Ollama ID."""
from pathlib import Path

env_path = Path('rag_service/.env')
if env_path.exists():
    content = env_path.read_text()
    # Fix the model name to include the full path
    new_content = content.replace(
        'LMSTUDIO_MODEL=qwen2.5-vl-7b',
        'LMSTUDIO_MODEL=qwen/qwen2.5-vl-7b'
    )
    if new_content != content:
        env_path.write_text(new_content)
        print("Fixed: LMSTUDIO_MODEL=qwen/qwen2.5-vl-7b")
    else:
        # Try alternate patterns
        lines = content.splitlines()
        new_lines = []
        found = False
        for line in lines:
            if line.startswith('LMSTUDIO_MODEL='):
                new_lines.append('LMSTUDIO_MODEL=qwen/qwen2.5-vl-7b')
                found = True
            else:
                new_lines.append(line)
        if found:
            env_path.write_text('\n'.join(new_lines))
            print("Fixed: LMSTUDIO_MODEL=qwen/qwen2.5-vl-7b")
        else:
            print("Could not find LMSTUDIO_MODEL line")
else:
    print(".env not found")
