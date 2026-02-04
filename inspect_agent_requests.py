import json
import sys
import os

# Add the directory to sys.path to import the module
agent_dir = r"e:\Office Projects\ai-agent\python-ai-agent\agents\2"
sys.path.append(agent_dir)

try:
    import builtins
    # Mocking necessary imports if the agent file fails to import due to dependencies
    # But usually it should be fine if we just want to import the variable.
    # Actually, the agent file imports app.config etc. which might be resolved relative to CWD e:\Office Projects\ai-agent\python-ai-agent
    # So we should add project root to path.
    project_root = r"e:\Office Projects\ai-agent\python-ai-agent"
    sys.path.append(project_root)
    
    # We can also just parse the file text to avoid import issues
    with open(os.path.join(agent_dir, "booking operations.py"), "r", encoding="utf-8") as f:
        content = f.read()
    
    import re
    match = re.search(r"AGENT_CONTEXT = json\.loads\(r'''(.+?)'''\)", content, re.DOTALL)
    if match:
        json_str = match.group(1)
        data = json.loads(json_str)
        requests = data.get("postman_requests", [])
        print(f"Found {len(requests)} Postman requests.")
        for req in requests:
            print(f"Name: {req.get('name')}")
            print(f"URL: {req.get('url')}")
            print(f"Start of Body: {str(req.get('body'))[:100]}")
            print("-" * 20)
    else:
        print("Could not find AGENT_CONTEXT in file.")

except Exception as e:
    print(f"Error: {e}")
