# Unbiased Parameter Estimation of Partially Observed Diffusions using Diffusion Bridges

This repository contains the implementation and results for the project *“Unbiased Parameter Estimation of Partially Observed Diffusions using Diffusion Bridges”*.  

The code is organized into source files, test cases, and numerical results.

---

## 📂 Project Structure

- **`bridge.py`**  
  Main implementation of the algorithm.

- **`PF_functions_def.py`**  
  Auxiliary functions used by the main code.

- **`Bridge_parallelization.py`**  
  Contains modified functions enabling parallel computing with the `multiprocessing` library.  
  Stores parameters for each Ibex run (each run identified by an ID).

- **`bridge_test.py`**  
  Collection of test routines covering each stage of the algorithm.

- **`test_result/Test.ipynb`**  
  Jupyter notebook displaying the numerical results presented in the paper.

- **`Kangaroo_data.txt`**  
  Observational data for the kangaroo population example.

- **`Girsanov approach matlab`**  
  Matlab code for the comparison with the Girsanov approach.

---

## 🧪 Tests

All major algorithmic components are tested in `bridge_test.py`.  
For reproducibility, additional test cases and results are shown in `test_result/Test.ipynb`.

---

## 📊 Data

- Kangaroo example observations: `Kangaroo_data.txt`  
- Due to memory restrictions, large datasets are hosted externally:  
  - **Full project data**: [Google Drive link](https://drive.google.com/drive/folders/1pQC05IfyalCOvboNE5VG6IKcbZ9rKWPL?usp=share_link)  
  - **Selected subset (for `Test.ipynb`)**: [Google Drive link](https://drive.google.com/drive/folders/1rqSLtdz3I_N2Q8J_FnK4YP9-gm9RAOSq?usp=share_link)

---

⸻

💻 Requirements
	•	Python 3.8+
	•	NumPy, SciPy, Matplotlib
	•	(Optional) Jupyter Notebook for exploring results
	•	Access to HPC cluster (e.g., Ibex) recommended for large-scale runs
