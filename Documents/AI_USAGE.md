# AI Usage & Attribution Statement

**Project:** Smart Image Classifier — Plant Disease Diagnosis  
**Selection Track:** GDG-USAR Technical Selection (Domain: AI / ML — Task 2)  
**Standard Compliance:** Item 03 & Item 07 of GDG-USAR Guidelines (*"Use documentation, tutorials, online resources, and AI tools responsibly. You must understand and explain your work."*)

---

I have created this project using vibe coding, AI and critical thinking.

## 1. AI Tools Utilized

- **Antigravity AI Assistant (Google DeepMind)**: Used as an interactive pair programmer for workspace inspection, dependency verification, and codebase packaging.
- **Large Language Models (Gemini / Claude)**: Employed for technical ideation, documentation structure, and drafting boilerplate utility code.

---

## 2. Scope of AI Involvement

| Area | Nature of AI Assistance | Human Oversight & Engineering Verification |
| :--- | :--- | :--- |
| **Model Architecture Exploration** | Suggested EfficientNet-B0 backbone with progressive MLP head ($1280 \rightarrow 512 \rightarrow 128 \rightarrow 38$) for parameter efficiency. | Verified dimension matching, dropout regularization rates, and PyTorch module compatibility with the checkpoint. |
| **Training Notebook Execution** | Initial training code was structured and run on Google Colab with GPU acceleration. | Monitored training convergence, verified loss curves, and validated Early Stopping triggers at Epoch 23. |
| **Metric Generation & Plots** | Scripted generation of high-resolution confusion matrix heatmaps and training progression curves. | Cross-checked accuracy (99.64%), precision, recall, and F1-scores against raw validation logs. |
| **Interactive Streamlit Web UI** | Drafted UI layout, tab components, and CSS styles for the drag-and-drop web demo. | Verified file upload handling, caching of model weights with `@st.cache_resource`, and real-time inference latency. |
| **Error & Failure Analysis** | Assisted in synthesizing botanical literature on foliar lesion morphology (Alternaria vs Corynespora). | Conducted image corruption experiments (blur, underexposure) and verified the resulting misclassification behavior. |

---

## 3. Human Understanding & Verification Pledge

All code, metrics, and technical rationales in this repository have been reviewed, understood, and validated:
- The underlying convolutional mechanics (depthwise separable convolutions, inverted residuals, squeeze-and-excitation blocks) are understood.
- The mathematical foundations of Cross-Entropy Loss, Softmax distributions, and confusion matrices are verified.
- The student is prepared to present, defend, and explain all design choices and code snippets during the GDG-USAR technical evaluation interview.
