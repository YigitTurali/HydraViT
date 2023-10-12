# HydraViT
A PyTorch implementation of the HydraViT model for multi-label disease classification from chest X-ray images.

## Description
HydraViT is an adaptive multi-branch transformer designed for the multi-label disease classification task using chest X-ray images. The model is trained and evaluated on the NIH Chest X-ray dataset.

## Usage
1. **Clone the repository:**
    ```bash
    git clone https://github.com/YigitTurali/HydraVit.git
2. **Navigate to the repository directory and install the required packages:**
   ```bash
   cd HydraViT
   pip install -r requirements.txt


## License
This project is licensed under the MIT License.

## Acknowledgements
The HydraViT model is based on the paper "HydraViT: Adaptive Multi-Branch Transformer for Multi-Label Disease Classification from Chest X-ray Images" by Şaban Öztürk, M. Yiğit Turalı, and Tolga Çukur.

Paper: HydraViT: Adaptive Multi-Branch Transformer for Multi-Label Disease Classification from Chest X-ray Images

Link: [arXiv:2310.06143](https://arxiv.org/abs/2310.06143)
Summary: Chest X-ray is a crucial diagnostic tool for identifying chest diseases due to its high sensitivity to lung abnormalities. However, diagnosing based on images is challenging due to the variability in pathology size and location, as well as the visual similarities and co-occurrence of different pathologies. Traditional convolutional neural networks (CNNs) face challenges due to their locality bias. The paper introduces HydraViT, a method that combines a transformer backbone with a multi-branch output module. This design enhances sensitivity to long-range context in X-ray images and focuses adaptively on essential regions. The multi-branch output module has an independent branch for each disease label, ensuring robust learning across different disease classes. Experiments show that HydraViT outperforms other methods in multi-label classification performance.
