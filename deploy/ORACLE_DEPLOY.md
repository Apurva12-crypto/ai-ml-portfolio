# Deploying the two services to Oracle Cloud (Always Free)

Only two projects are services: the **PII redaction API** (04) and the **fraud scoring API** (02). The RUL, retrieval and synthetic-data projects are batch/analysis work and need no server.

I could not log in to your Oracle account, so these steps are for you to run. Docker build was **not tested** from my workspace.

## 1. Create the VM (console.cloud.oracle.com)
- Compute -> Instances -> Create. Shape: **VM.Standard.A1.Flex** (Ampere ARM, 1-2 OCPU, 6-12 GB is inside Always Free) or VM.Standard.E2.1.Micro.
- Image: Ubuntu 22.04. Upload your SSH public key.
- Networking: open ingress TCP **22, 8000, 8001** in the subnet security list (or put a reverse proxy on 443).

## 2. On the VM
```bash
sudo apt-get update && sudo apt-get install -y docker.io docker-compose-v2 git
sudo usermod -aG docker $USER && newgrp docker
# Ubuntu images also have a host firewall:
sudo iptables -I INPUT -p tcp -m multiport --dports 8000,8001 -j ACCEPT
git clone https://github.com/<you>/<your-portfolio-repo>.git && cd <your-portfolio-repo>
```
## 3. Fraud model file
`02_paypal_fraud/results/lgbm.txt` and `meta.json` must come from your **real Kaggle run**. Download them from the notebook output and put them in that folder before building. Do not deploy a model trained on the synthetic smoke-test data.

## 4. Run
```bash
docker compose up -d --build
curl localhost:8000/health && curl localhost:8001/health
curl -X POST localhost:8000/redact -H 'content-type: application/json' -d '{"text":"Mail me at a@b.com"}'
```
Public URLs: `http://<vm-public-ip>:8000/docs` and `:8001/docs` (FastAPI's interactive docs).

## 5. Before you put the link on a resume
- Add an API key header check or restrict by IP; the fraud endpoint exposes a model.
- Never send real personal data to the demo PII endpoint.
- Free-tier VMs can be reclaimed if idle; keep the repo's README the source of truth.
