- install calvin in a new, separate environment
- in the original (manten) environment: run `manten.scripts.agent_proxy`
- use `manten_calvin_agent_proxy_client.py` as a `CustomAgent` to evaluate calvin using their instructions in the new environment

notes:

- if you have problems while installing calvin with pyhash, remove it from the requirements.txt and install via conda, it installs the binary directly
