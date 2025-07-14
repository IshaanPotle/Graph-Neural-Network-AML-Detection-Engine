# Graph Neural Network AML Detection Engine

A production-ready Anti-Money Laundering (AML) detection system using Graph Neural Networks (GNNs) to analyze cryptocurrency transaction networks in real-time.

---

## ⚠️ Large Dataset & Backup Files
Some files required for this project (datasets and backups) are too large for GitHub and are hosted on Hugging Face.

**Download them from:**
https://huggingface.co/IshaanPotle27/elliptic-aml-model

Place the downloaded files in the appropriate directories as described below.

---

## Features
- Real-time fraud detection using GraphSAGE, GAT, and Temporal Graph Networks
- FastAPI backend and Streamlit dashboard for monitoring and explainability
- Scalable, production-ready Dockerized architecture
- Enterprise-grade security, monitoring, and compliance features

## Quick Start
1. **Clone the repository:**
   ```bash
   git clone https://github.com/IshaanPotle/Graph-Neural-Network-AML-Detection-Engine.git
   cd Graph-Neural-Network-AML-Detection-Engine
   ```
2. **Download large datasets and backups from Hugging Face** ([link](https://huggingface.co/IshaanPotle27/elliptic-aml-model)) and place them in the correct folders (see above).
3. **Run the Docker setup:**
   ```bash
   ./scripts/docker-setup.sh
   ```
4. **Access the services:**
   - Dashboard: http://localhost:8501
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

## Documentation
- **Docker & Deployment:** See `README-Docker.md`
- **Improvements & Features:** See `README-Improvements.md`

---

For more details, see the full documentation in this repository. 