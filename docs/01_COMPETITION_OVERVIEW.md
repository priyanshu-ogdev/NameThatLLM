# Name That LLM: Competition Overview & Requirements

## 1. Executive Summary

**Name That LLM** is an active machine learning code competition hosted on Kaggle. It challenges competitors to tackle the problem of **AI authorship attribution and stylometric fingerprinting**: given an anonymous, short paragraph of generated text (typically 40 to 250 words), participants must build classification models to accurately identify which of **18 leading frontier large language models** authored the text.

The challenge reflects real-world frontiers in AI safety, watermarking, security auditing, benchmark integrity, and intellectual property attribution. Because modern frontier models are trained on overlapping open-web corpora and fine-tuned for helpfulness, surface-level topical differences are minimal. Models must instead be distinguished by their subtle syntactic signatures, formatting idiosyncrasies, punctuation cadences, alignment tendencies, and vocabulary biases.

---

## 2. Competition Metadata & Timeline

| Parameter | Details |
| :--- | :--- |
| **Competition Name** | Name That LLM |
| **Platform** | Kaggle ([Competition Page](https://www.kaggle.com/competitions/name-that-llm)) |
| **Competition Type** | Code Competition (Synchronous execution on hidden test data) |
| **Total Prize Pool** | **$50,000 USD** |
| **Prize Distribution** | **10 equal awards of $5,000 each** (1st through 10th place on Private Leaderboard) |
| **Launch Date** | October 14, 2026 |
| **Entry & Team Merger Deadline** | January 6, 2027, 23:59 UTC (3:59 PM PST) |
| **Final Submission Deadline** | **January 13, 2027, 23:59 UTC** (3:59 PM PST) |
| **Max Team Size** | 5 members |
| **Daily Submission Limit** | 5 submissions per day |
| **Final Submissions Selection** | 2 submissions for final Private Leaderboard scoring |

> **Timeline Precaution:** Organizer API metadata states `2027-01-14T23:59:00.817Z`, whereas public frontend documents state January 13, 2027 at 23:59 UTC. Teams must plan all final submissions for January 13, 2027 to avoid missing deadlines due to discrepancies.

---

## 3. Evaluation Metric: Balanced Accuracy

Submissions are evaluated on **Balanced Accuracy Score** across all 18 target model classes.

### Mathematical Formulation
Balanced accuracy is defined as the macro-average of recall obtained on each class:

$$\text{Balanced Accuracy} = \frac{1}{K} \sum_{k=1}^{K} \text{Recall}_k = \frac{1}{K} \sum_{k=1}^{K} \frac{TP_k}{TP_k + FN_k}$$

Where:
- $K = 18$ is the total number of candidate LLMs.
- $\text{Recall}_k$ is the true positive rate (sensitivity) for model class $k$.
- $TP_k$ is the count of passages correctly predicted as model $k$.
- $FN_k$ is the count of passages authored by model $k$ that were incorrectly classified.

### Strategic Implications for Competitors
1. **Equal Weighting Across Classes:** Unlike standard multi-class accuracy, each of the 18 models contributes exactly $\frac{1}{18} \approx 5.56\%$ to the overall score.
2. **Combating Class Imbalance:** If the private test set has varying numbers of samples per model, predicting frequent models at the expense of rarer models severely degrades the balanced score.
3. **Threshold & Prior Calibration:** Classifiers that predict probabilities or decision margins must calibrate decision thresholds to prevent dominating classes (like `claude-opus-4-6` or `gemini-3.1-pro-preview`) from suppressing low-recall classes (such as `claude-fable-5` or `glm-5.3-flash`).
4. **Random Guessing Baseline:** Pure random guessing or majority-class baseline yields an expected balanced accuracy of:
   $$\frac{1}{18} \approx 0.0556 \text{ (5.56\%)}$$

---

## 4. Code Competition Submission & Runtime Constraints

As a Kaggle Code Competition, submissions are evaluated automatically by rerunning notebooks in an offline container against hidden test data.

### Infrastructure & Hardware Parameters
- **Runtime Limit:** Maximum of **9 hours** total execution time per submission.
- **Accelerator Options:**
  - **CPU-only:** High-core Intel Xeon environment (ideal for high-dimensional linear models, TF-IDF feature unions, SVMs).
  - **GPU Accelerated:** Typically **Tesla T4 x2 (16 GB VRAM each)** or P100 (16 GB VRAM).
- **Internet Access:** **Strictly DISABLED** during scoring runs. All pre-trained weights, tokenizers, custom packages, and offline Python wheels must be bundled into Kaggle Datasets attached to the notebook.
- **Submission Output:** The submission file must be saved to:
  ```
  /kaggle/working/submission.parquet
  ```
  *(Note: The submission format is Parquet, not CSV).*
- **Input Data Location:** The evaluation runner executes against:
  ```
  /kaggle/input/competitions/name-that-llm/test.csv
  ```
  During interactive editing, `test.csv` contains a small dummy placeholder (3 sample rows) to verify schema and pipeline execution. During hidden scoring, `test.csv` is dynamically swapped for the full hidden test set of approximately **4,000 text passages**.

---

## 5. External Data & Resource Compliance (Rule 6)

The competition allows external pretrained models and auxiliary datasets subject to Kaggle's standard openness criteria:
1. **Public Availability:** Any external model weights or datasets used must be freely accessible to all participants at no cost and posted publicly on the competition forums or Kaggle Datasets.
2. **License Compliance:** Models such as Apache 2.0 (Qwen family), MIT (BGE models), and community response corpuses under Creative Commons (e.g. `olegzholobov/name-that-llm-response-corpus`) comply.
3. **No Paid APIs During Competition:** Solutions cannot make external network calls to paid proprietary APIs (OpenAI, Anthropic, Google) during the notebook evaluation.
