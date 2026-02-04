Project Summary - python-ai-agent (assistant-ai-server)
Purpose: FastAPI runtime + agent template system for LLM tools, realtime chat, and integrations.

Key paths:
- main.py: FastAPI app, HTTP/WS endpoints, runtime logic.
- agent_template.py.j2: primary agent template used for generated agents.
- templates/agent_template.py.j2: synced copy used by the builder.
- runtime.py: shared runtime helpers and agent execution utilities.

Runtime:
- Server path: /home/abdullah/hotels-task/assistant-ai
- Agents generated at: /home/abdullah/hotels-task/assistant-ai/agents/{device_id}/{agent}.py
- Agent context stored at: /home/abdullah/hotels-task/assistant-ai/data/{device_id}/{agent}/context.json

Server access:
- Host: 191.101.80.47
- User: root
- SSH key: /home/mohamed-samer/Downloads/entaai_deploy_key
- Project path: /home/abdullah/hotels-task/assistant-ai

Core endpoints:
- POST /agents, PUT /agents/{name}: create/update agents from builder.
- POST /agents/{name}/run: text chat (HTTP).
- WS /ws/agents/{name}/run: streaming text chat.
- WS /ws/realtime/agents/{name}/run: realtime text+audio.
- POST /images/white-background: background whitening for avatars.

Realtime notes:
- Uses OpenAI Realtime for streaming audio when enabled.
- audio_transcript in done payload mirrors final text output for UI display.

Build/deploy notes:
- FastAPI served by uvicorn.
- Requires OpenAI keys and any integration keys in .env.

Flow automation/selection handling:
- app/runtime.py: regex-based option parsing + numeric selection helpers; _auto_postman_from_flow_selection can pick the next Postman endpoint and map selections into query/body.
- app/main.py: intercepts option selections in WS flows and may call runtime_helpers._auto_postman_from_flow_selection before running the agent.
- agent_template.py.j2: generated agent includes its own option-selection logic, endpoint choice via LLM, and Postman auto-calls (can diverge from runtime helpers).
