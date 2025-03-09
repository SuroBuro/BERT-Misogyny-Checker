# 🚀 Multi-Level Misogyny Detection Using BERT

## 📌 Overview
This project fine-tunes **BERT-based models (BERT & HateBERT)** for **multi-level misogyny detection**. It classifies online text into three hierarchical levels:  
- **Level 1**: **Binary classification** (Misogynistic / Non-Misogynistic)  
- **Level 2**: **Types of misogyny** (Pejorative, Treatment, Derogation, etc.)  
- **Level 3**: **Contextual categories** (Personal Attack, Counter Speech, None, etc.)  

The model also extracts **attention scores** to **highlight the most important words** contributing to the classification.  

---

## 📌 1. Model Architecture & Methodology

### **1.1 Base Model**
- **BERT-base-uncased** *(or HateBERT for improved performance)*
- **Multi-Task Learning (MTL)**: Separate classification heads for each level
- **Loss Function**: CrossEntropyLoss applied at all three levels
- **Optimizer**: AdamW with a learning rate of `2e-5`
- **Dataset**: [EACL 2021 Expert Annotated Misogyny Dataset](https://aclanthology.org/2021.eacl-main.114.pdf)

### **1.2 Training Pipeline**
1. **Data Preprocessing**  
   - **Clipped value ranges** to fit label mappings instead of using `SimpleImputer` for NaN values.
   - Tokenized text using **BertTokenizer**.
   - Sequences **truncated/padded** to `max_length = 512`.
   
2. **Model Fine-Tuning**  
   - Trained for **4 epochs** with **batch size = 8**.
   - **Weighted loss** to counter dataset imbalance.
   - Fine-tuned **HateBERT** for better performance.

3. **Explainability Features**  
   - Extracted **BERT’s attention scores**.
   - Highlighted **top 3 words** influencing classification.  
   

---

## 📌 2. Evaluation Metrics

### **2.1 Model Performance**
| **Metric**   | **Level 1 (Binary)** | **Level 2 (Multi-Class)** | **Level 3 (Multi-Class)** |
|-------------|----------------------|---------------------------|---------------------------|
| **Accuracy** | 98.61%               | 97.83%                    | 100%                      |
| **Precision** | 98.75%               | 97.79%                    | 100%                      |
| **Recall** | 98.61%               | 97.83%                    | 100%                      |
| **F1-Score** | 98.65%               | 97.75%                    | 100%                      |

### **2.2 Explainability**
- Extracted **attention scores** from the last transformer layer.
- **Highlighted words** most relevant to the classification.


---

## 📌 3. Challenges & Solutions
| **Challenge** | **Solution Implemented** |
|--------------|--------------------------|
| **Overfitting due to small dataset** | Used **BERT** instead, applied **dropout & weight decay** |
| **Model returning only 2 outputs instead of 3** | Ensured correct loading using `AutoModelForSequenceClassification` |
| **Large Model File (>100MB) Not Uploading to GitHub** | Used **Git LFS** & stored the model on **Google Drive** |
| **Debugging API Issues (Tensor/String Mismatch)** | Corrected logits extraction using `outputs["logits"]` |
| **Visualizing Model Performance** | Improved **bar graphs & line graphs** in Matplotlib |

---

## 📌 4. Future Improvements
- **Fine-tune the model on a larger dataset**
- **Improve sarcasm & implicit bias detection**
- **Deploy the Flask API with a chatbot-style UI**
- **Optimize for real-time moderation in social media platforms**

---

## 📌 5. How to Use the Model

### **5.1 Running the Flask API**
1. **Clone the Repository**
   ```sh
   git clone https://github.com/SuroBuro/BERT-Misogyny-Checker.git
   cd BERT-Misogyny-Checker
