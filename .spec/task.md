## triton-deploy

### Overview

In this project, you will simulate a real-world deployment of a **computer vision model** using **NVIDIA Triton Inference Server**. The goal is to deploy one or more pre-trained models, optimize inference performance, and integrate DevOps practices with containerization and monitoring. The focus is on achieving efficient serving, logging, and performance tracking in a Docker-based environment.

### Role Play

You are part of a DevOps team at a manufacturing company. Your task is to deploy a computer vision model (for example, defect detection or image classification) into a production-like environment. The challenge is to serve the model efficiently, monitor performance, and make sure the deployment is reliable and scalable within a Docker-based setup.

### Learning Objectives

- Deploy pre-trained computer vision models using NVIDIA Triton Inference Server.
- Optimize inference with TensorRT and ONNX.
- Use Docker to containerize the environment for reproducible deployment.
- Monitor latency, throughput, and (on a GPU) GPU usage with Prometheus and Grafana.
- Log model requests and track inference history.
- Manage multiple versions of a model in Triton.

### Instructions

#### Model Serving

- Set up **NVIDIA Triton Inference Server** in Docker.
- Deploy at least **two pre-trained models** (e.g., ResNet50 for classification and YOLO for detection).
- Enable features such as **dynamic batching** and **GPU optimization**.
- Provide inference endpoints accessible through REST or gRPC.

#### Performance Optimization

- Convert at least one pre-trained model to **TensorRT** or **ONNX** for optimization.
- Benchmark inference before and after optimization.
- If you do not have an NVIDIA GPU, use the **ONNX** backend, which runs Triton and ONNX Runtime on CPU; report CPU metrics in that case. (TensorRT specifically requires an NVIDIA GPU.)
- Compare performance metrics such as latency and throughput.

#### DevOps with Docker

- Containerize the full setup with **Docker**.
- Provide a `Dockerfile` and scripts to build and run the project locally.
- Ensure reproducible deployments on any machine (GPU recommended; the ONNX backend runs CPU-only, as noted above).

#### Monitoring & Logging

- Integrate **Prometheus** for collecting metrics.
- Configure **Grafana dashboards** to visualize performance (latency, throughput, and GPU usage when running on a GPU).
- Log model requests and responses for debugging and analysis.

#### Scalability & Versioning

- Serve **multiple versions** of at least one computer vision model.
- Document how version switching and model reloading works in Triton.

#### Bonus

- Add **alerting rules** in Prometheus for high latency or failed requests.
- Export inference logs to CSV for offline analysis.

#### Dataset

- Use pre-trained weights for models (e.g., ImageNet-pretrained ResNet, COCO-pretrained YOLO).
- Include at least 10 sample images in `client/samples/` for testing inference.

#### Code Quality

- Keep code modular: separate deployment scripts, monitoring configs, and client code.
- Provide clean and reusable Python scripts for inference requests.
- The **README.md** should explain:
  - The project's purpose and scope
  - How to set up Triton with Docker
  - How to run inference
  - How to monitor with Prometheus + Grafana
  - Repository structure

### Submission Structure

```
triton-deploy/
├── models/
│   ├── resnet50/
│   └── yolov5/
│
├── deployment/
│   ├── docker/
│   │   └── Dockerfile
│   └── run_triton.sh
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana-dashboard.json
│
├── client/
│   ├── inference_client.py
│   └── samples/
│       ├── img1.jpg
│       └── img2.jpg
│
├── requirements.txt
├── README.md
└── .env (optional)
```

### Tips

- Start by deploying one pre-trained model before adding a second.
- Use Docker volumes to manage models in the Triton container.
- Benchmark inference with different batch sizes to see optimization effects.
- Keep monitoring simple but informative.

### AI Prompts For Learning

Use these to build the understanding this project needs. They ask the AI to explain concepts and question you, not to write your solution. If the AI starts giving you code, tell it to stop and ask you a question instead.

- "Explain the difference between training a model and serving it for inference, then ask me what new problems appear only at serving time."
- "Explain what Triton Inference Server does, then quiz me on how it can serve several models and frameworks at once."
- "Explain dynamic batching, then ask me to predict what it does to latency versus throughput under load."
- "Ask me what ONNX and TensorRT each change about a model, and which one I can use without a GPU."
- "Quiz me on why model versioning and rollback matter in production, and how Triton handles them."

**A note on using AI:** If AI solves this, AI gets smarter, not you. Every problem you push through yourself is what rewires your brain and builds the skill. Let it think for you and that growth never happens. Use AI to understand, then do the thinking yourself, because your intelligence is the one worth developing.






========== AUDIT ==========





#### General

##### Submission Structure

```
triton-deploy/
├── models/
│   ├── resnet50/
│   └── yolov5/
│
├── deployment/
│   ├── docker/
│   │   └── Dockerfile
│   └── run_triton.sh
│
├── monitoring/
│   ├── prometheus.yml
│   └── grafana-dashboard.json
│
├── client/
│   ├── inference_client.py
│   └── samples/
│       ├── img1.jpg
│       └── img2.jpg
│
├── requirements.txt
├── README.md
└── .env (optional)
```

###### Does the project follow the structure described in the subject `Submission Structure` section?

##### Play the Role of a Stakeholder

The auditor plays the role of a teammate reviewing the learner's deployment before it ships. Spend 5-10 minutes asking the learner to explain their own setup. The goal is to confirm the learner understands what they built, not just that it runs. A learner who wrote the project can answer these; one who only generated it cannot.

1. Walk me through what happens to an image from the client request to the prediction coming back from Triton.
2. What did optimizing with ONNX or TensorRT change, and how did you measure the difference?
3. What does dynamic batching do, and when does it help or hurt?
4. How does Triton serve and switch between multiple model versions?
5. Show me a metric in Grafana and explain what it would look like if the server were overloaded.

###### Can the learner explain the serving pipeline and the role of Triton without reading from the code?

###### Can the learner correctly interpret a latency or throughput metric and reason about it during the audit?

###### Does the learner demonstrate genuine understanding of model serving, optimization, and monitoring, rather than reciting generated code?

##### Setup

###### Does the `README.md` explain the project's purpose, how to set up Triton with Docker, how to run inference, how monitoring is configured, and the repository layout?

###### Does `requirements.txt` list the necessary libraries (for example `tritonclient`, `onnxruntime`, `prometheus_client`)?

###### Are all required files present and correctly placed (`models/`, `deployment/docker/Dockerfile`, `deployment/run_triton.sh`, `monitoring/prometheus.yml`, `monitoring/grafana-dashboard.json`, `client/inference_client.py`, `client/samples/`)?

##### Model Serving

###### Is NVIDIA Triton Inference Server set up and running inside Docker?

###### Are at least two pre-trained computer vision models (for example ResNet50 and YOLO) deployed?

###### Are inference endpoints accessible via REST or gRPC?

###### Are Triton features like dynamic batching or GPU optimization enabled and configured?

###### Can the client script send images and receive predictions successfully?

##### Performance Optimization

###### Is at least one model optimized with TensorRT or ONNX before deployment?

###### Are benchmarks performed both before and after optimization, comparing latency and throughput?

###### Can the learner explain the before/after benchmark result, including cases where the optimized model shows little or no speedup (e.g. on CPU or with small batches)?

**Note for Auditors:** There is no fixed improvement bar. A correct optimization (ONNX conversion, dynamic batching, etc.) may show little or no speedup on CPU or for some model/batch sizes; a sound methodology with a flat result still passes as long as the learner can explain it.

##### DevOps with Docker

###### Is the full system containerized with a working `Dockerfile` and `run_triton.sh` script?

###### Can the project be built and run reproducibly on any machine (GPU recommended; CPU works with the ONNX backend)?

##### Monitoring and Logging

###### Is Prometheus integrated to collect metrics (latency, throughput, and GPU usage when on a GPU)?

###### Is a Grafana dashboard provided to visualize the metrics?

###### Are inference requests and responses logged for debugging?

##### Scalability and Versioning

###### Are multiple versions of at least one model served in Triton?

###### Is version switching or model reloading demonstrated and documented?

##### Repository and Code Quality

###### Is the code modular (deployment scripts, monitoring configs, and client separated logically)?

###### Are the functions well-documented and reusable (for example the inference client)?

###### Do the scripts run without errors (for example `python client/inference_client.py`)?

#### Bonus

###### +Are alerting rules in Prometheus configured for high latency or failed requests?

###### +Are inference logs exported to CSV for offline analysis?