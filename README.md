# ConstructNLPfromScratch

## Introduction
ConstructNLPfromScratch is a project aimed at building a Natural Language Processing (NLP) system from the ground up. This repository provides a step-by-step implementation to help learners and developers understand the core concepts of NLP, including tokenization, part-of-speech tagging, and basic language modeling, without relying on high-level libraries.
This project is altered based on Minimind Project:  https://github.com/jingyaogong/minimind 

## Features
- Custom tokenization and sentence splitting
- Basic part-of-speech tagging implementation
- Simple language model using n-grams
- Extensible framework for adding new NLP components

## Installation
To get started with ConstructNLPfromScratch, follow these steps:

1. Clone the repository:
   ```
   git clone https://github.com/arthurpanhku/ConstructNLPfromScratch.git
   cd ConstructNLPfromScratch
   ```
2. Ensure you have Python 3.8+ installed.
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
4. Run the setup script:
   ```
   python setup.py
   ```

## Usage
To use the project, follow these steps:

### 1. Prepare Translation Data

Create a JSONL file (e.g., translation.jsonl) containing source and target language dialogue data. The format should resemble that used in SFTDataset or DPODataset. For example:
```json
{"conversations": [{"role": "user", "content": "Hello, how are you?"}, {"role": "assistant", "content": "Hola, ¿cómo estás?"}]}
{"conversations": [{"role": "user", "content": "Good morning!"}, {"role": "assistant", "content": "¡Buenos días!"}]}
```
Ensure each line is a JSON object with a conversations field, containing a list of role-content pairs.
Use the source language (e.g., English) as user input and the target language (e.g., Spanish) as the assistant output.

### 2. Configure the Training Environment

Ensure all necessary dependencies are installed (based on run.sh and requirements.txt, including pytest and other libraries).
Configure parameters according to train_sft.py or train_dpo.py, adjusting data_path to point to your translation.jsonl file. 

For example:

```bash
python train_sft.py --data_path ./translation.jsonl --epochs 1 --batch_size 32 --learning_rate 5e-5
```

Use SFTTrainer for supervised fine-tuning (SFT) as it supports dialogue data and generates predictions.
Ensure max_seq_len and max_new_tokens are sufficiently long for translation tasks (e.g., 512 and 1024).

### 3. Train the Model

Run the run.sh script or execute train script individually:

```bash
./run.sh
```
The script will run tests (test_attention.py and test_dpo.py) and perform pretraining, SFT, LoRA, and DPO training.
After training, the model checkpoint will be saved to out/sft_512.pth (assuming dim=512).

### 4. Generate Translations

Use the SFTTrainer.get_predictions method to generate translations:
Modify train_sft.py or create a new script to call trainer.get_predictions.
Provide an input message list, for example:

```python
messages_lst = [{"role": "user", "content": "Hello, how are you?"}]
predictions = trainer.get_predictions(messages_lst)
print(predictions)  # Output the translation result
```

The model will generate the target language translation based on the training data.


## Contributing
Contributions are welcome! Please follow these steps:
- Fork the repository.
- Create a new branch for your feature: `git checkout -b feature-name`.
- Commit your changes: `git commit -m "Add feature-name"`.
- Push to the branch: `git push origin feature-name`.
- Submit a pull request.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact
For questions or feedback, please open an issue in the repository or contact the maintainer at u3638376@connect.hku.hk
