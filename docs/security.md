# Security

This server is built for a machine you control and a network you trust. Out of the box it has no password and no encryption, so
read this page before you open its port to anyone else. This page covers the API key, what the key does not protect, and how
to reach the server safely over an untrusted network.

## Protect the server with an API key

By default the server needs no password: anyone who can reach port 8000 can use it. To require a key, set `API_KEY` when you
start the server. The key is any secret string; this makes a random one:

```bash
export API_KEY=$(openssl rand -hex 32)
echo "$API_KEY"                      # write it down: every client needs it
scripts/run.sh qwen3.8-flash-next    # or: scripts/install.sh qwen3.8-flash-next
```

The server now answers only requests that carry the key as a bearer token. Without it (or with a wrong key) you get
`401 Unauthorized`:

```bash
curl -s http://localhost:8000/v1/models
# {"error":"Unauthorized"}

curl -s http://localhost:8000/v1/models -H "Authorization: Bearer $API_KEY"
# {"object":"list","data":[{"id":"qwen3.8-flash-next", ...
```

In a client, enter the key where it asks for an API key. For example with the Python `openai` package:

```python
from openai import OpenAI

client = OpenAI(base_url="http://<spark>:8000/v1", api_key="<your key>")
reply = client.chat.completions.create(
    model="qwen3.8-flash-next",
    messages=[{"role": "user", "content": "Hello"}],
)
print(reply.choices[0].message.content)
```

The test tools in `tools/` read the key from the environment: `LLM_API_KEY="$API_KEY" python3 tools/answer_check.py qwen3.8-flash-next`.

To change or remove the key, start the server again with the new value (or without `API_KEY`) and the same `PROFILE`:
`scripts/run.sh <model>` replaces the running container. The key stays in the container's settings, so it survives a reboot.

What the key does not do:

- It protects only the API paths (`/v1/...`). `/health` and `/metrics` stay open to everyone who can reach the port; they show
  that the server is up and how busy it is, not prompts or answers.
- It is one shared key for all clients, and anyone who can run `docker inspect` on the Spark can read it.
- The connection is plain HTTP, so the key and everything you send travel unencrypted. On a network you do not fully trust,
  do not open port 8000 to it: reach the server through an SSH tunnel (`ssh -L 8000:localhost:8000 <user>@<spark>`, then use
  `http://localhost:8000/v1`) or put a reverse proxy with HTTPS in front.

## Checklist

- [ ] `API_KEY` is set and written down somewhere safe (not in a repository, not in a chat).
- [ ] Port 8000 is not reachable from the internet or from networks with people you do not trust (router or firewall).
- [ ] If clients connect over a network you do not control: SSH tunnel, VPN or HTTPS reverse proxy, not plain HTTP.
- [ ] Nothing else important runs on the Spark: the profiles use most of the memory, and running out can freeze the machine
  (see the [README](../README.md#limitations-and-responsibility)).
