# GradES

Official implementation of **GradES** - a gradient-based selective training method that dynamically freezes converged modules during fine-tuning to achieve significant computational savings without sacrificing model performance.

## 📄 Paper
[Paper Title] - Accepted at [Conference Name 2025]

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/GradES.git
cd GradES

# Install dependencies
pip install -r requirements.txt

# Run GradES fine-tuning
python train_grades.py --model qwen3-14b --method lora --threshold 0.025

📊 Key Results

    40-50% computational savings compared to standard fine-tuning

    Maintains or improves model performance across multiple benchmarks

    Tested on Qwen3, Phi4, Llama-3.1, and Mistral models (0.6B to 14B parameters)

📖 Citation

If you find GradES useful in your research, please cite:

@inproceedings{grades2025,
  title={GradES: Gradient-based Early Stopping for Efficient Fine-tuning of Large Language Models},
  author={Your Name and Coauthor Names},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2025},
  pages={xxx--xxx}
}

📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

🤝 Contributing

We welcome contributions! Please see CONTRIBUTING.md for details.

📧 Contact

For questions, please open an issue or contact [your-email@institution.edu]
