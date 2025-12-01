# NeuroBioMark-AI

# 🧠 NeuroBioMark AI
A Multi-Agent MRI Biomarker Analysis System Powered by Google’s Agent Development Kit

NeuroBioMark AI is an enterprise-grade multi-agent system designed to analyze structural MRI data, compute biomarker indices, generate research-grade reports, and provide an intelligent biomedical assistant powered by Gemini 2.5 Flash.

Built using Google’s Agent Development Kit (ADK), the platform orchestrates multiple specialized agents—data ingestion, biomarker computation, risk estimation, longitudinal tracking, and LLM-powered interpretation—to deliver a seamless MRI analysis workflow for biomarker research.

This project was developed as part of the Google 5-Day Agents Intensive Capstone Project.


# 🚀 Key Features
## 🔹 1. Multi-Agent System

The system orchestrates several specialized agents:

### Agent	Responsibility

| Agent                   | Responsibility                                                        |
| ----------------------- | --------------------------------------------------------------------- |
| **Intake Agent**        | Loads DICOM/NIfTI MRI data and prepares structured inputs             |
| **Biomarker Agent**     | Computes hippocampal volume, ventricle volume, GM/WM ratios, z-scores |
| **Risk Agent**          | Computes non-diagnostic risk tiers using composite MRI indices        |
| **Report Agent**        | Generates research-grade narratives and PDF reports                   |
| **LLM Assistant Agent** | Uses Gemini 2.5 Flash to answer MRI biomarker questions               |

All agents communicate through the ADK’s orchestration layer — with tools, memory, and context sharing.

## 🔹 2. MRI Biomarker Computation

The platform computes:

- Hippocampal volume

- Ventricular volume

- GM volume

- WM volume

- Intracranial volume (ICV)

- **Z-scores** relative to a reference population

- **Composite indices:**

   -  Ventricle/Hippocampus ratio

   -  Atrophy index

   -  GM/WM ratio

## 🔹 3. AI-Generated Biomarker Report (PDF)

NeuroBioMark AI produces a clean PDF report including:

- Subject/timepoint metadata

- Biomarker z-scores

- Composite indices

- Risk estimation (non-diagnostic)

- Longitudinal progression summary

- AI-generated narrative interpretation

- Research disclaimer


## 🔹 4. Research Assistant Powered by Gemini 2.5

A built-in chat assistant provides:

- MRI biomarker interpretation

- Research explanations

- Risk factor breakdowns

- Pathophysiology insights

- Longitudinal reasoning

All strictly **non-diagnostic**.

## 🔹 5. Longitudinal Tracking

The system detects trends across timepoints:

- Atrophy progression

- Ventricular expansion

- GM/WM changes

- Cognitive label transitions (CN → MCI → AD)


## 🧩 Architecture Overview

### High-Level System Flow

<img width="359" height="347" alt="image" src="https://github.com/user-attachments/assets/e2d3889f-988f-440f-8079-5d5fadfb0603" />


### Technologies

- Python 3.11

- Streamlit (UI)

- Google ADK (Agents + Tools + Orchestration)

- google-genai 1.52.0

- FPDF 2.x (unicode PDF)

- NumPy, Pandas, Matplotlib



## 📦 Project Structure

<img width="616" height="404" alt="image" src="https://github.com/user-attachments/assets/bcd36ef9-babd-4149-a9f4-10bee99f9498" />

## 🔑 Setup Instructions

1. ***Clone the repository***

- `git clone git@github.com:Pauligtech/NeuroBioMark-AI.git`
- `cd NeuroBioMark-AI`

2. ***Create virtual environment***

- `python3 -m venv .venv`
- `source .venv/bin/activate`

3. ***Install dependencies***

- `pip install -r requirements.txt`

4. ***Add your Gemini key***

    Create a file:

- `.gemini_api_key`

    Inside it:
- `YOUR_GEMINI_KEY_HERE`

    Or export environment variable:

- `export GEMINI_API_KEY="your-key"`

### ▶️ Run the App

- `streamlit run app.py`


### 🧪 Sample Input Data

  Put DICOM/NIfTI into:

- `sample_input.dcm`

Or use your own MRI data.




