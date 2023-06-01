# README

This repository is where we implemented the "" paper.

### **Datasets and Preprocessing**

---

For training the detector, we used LibriSpeech (train-clean-100) and Atis data. For Atis data, we obtained speech files from TTS published on espnet.

We then obtained ASR results through google, whisper, and conformer speech recognizers.  Only the recognition results for LibriSpeech (train-clean-100) are published in the "" folder for each recognizer version.

- LibriSpeech data source : [https://www.openslr.org/12](https://www.openslr.org/12)
- Atis data source : [https://www.kaggle.com/datasets/hassanamin/atis-airlinetravelinformationsystem?select=atis_intents_train.csv](https://www.kaggle.com/datasets/hassanamin/atis-airlinetravelinformationsystem?select=atis_intents_train.csv)
- TTS sources used : [https://espnet.github.io/espnet/notebook/espnet2_tts_realtime_demo.html](https://espnet.github.io/espnet/notebook/espnet2_tts_realtime_demo.html)
- Google Speech Recognizer Source : [https://cloud.google.com/speech-to-text](https://cloud.google.com/speech-to-text)
- Whisper recognizer source : [https://github.com/openai/whisper](https://github.com/openai/whisper)
- Conformer recognizer source : [https://zenodo.org/record/4604066#.ZHa0hOxBweY](https://zenodo.org/record/4604066#.ZHa0hOxBweY)

Data drive link : [ASR_for_Train](https://sogang365-my.sharepoint.com/:f:/g/personal/yeen214_o365_sogang_ac_kr/EpYzziwG-WRKtDSmjJFxFrsBGeeXWzZuuTNMaxmFuNRgsw?e=mEqdQO)

### **Requirements**

---

To build the environment, run the following code

```python
pip install -r requirements.txt
```

### **Training**

---

**1) Detector**

- run the following code. However, before running the code, you need to put **train_sample.txt** into the **Detector_training/data** directory. The txt data consists of **'{original_text}_{ASR}'** per line.

```python
python3 main_train.py --config_file config.json
```

**2) Corrector**

- Corrector training is also done by placing the **corrector_train.txt** file in the **Corrector_training/data** directory and running the code below. The process is that data preprocessing happens in **preare_dataset.py** and training happens in **pretrain.py**

```python
python3 prepare_dataset.py 
```

- after preparing dataset, run the following code to run **pretrain.py**

```python
python3 pretrain.py --dataset t5
```

### Inference

---

**1) Detector** 

Put the trained model in the **Inference/Detector/models** directory before inference.

If you check the [test.py](http://test.py/) file, you can infer about Librispeech test, IC, and ER.

```python
cd Detector
python3 test.py
```

**2) Corrector** 

Put the trained model in the **Inference/Corrector/model** directory before inference.

Utilize the output from the detector and put it into the corrector.

```python
python3 T5_test.py --model model --batch 32 --data_dir 'input_data_example/er/final_er_test.csv' --max_src_len 512 --max_trg_len 512
```

Representative Detector & Corrector checkpoint link : [D&C_checkpoint](https://sogang365-my.sharepoint.com/:f:/g/personal/yeen214_o365_sogang_ac_kr/EhHvKsw5vlNEojFIzP9kEVUB8t9CDP5TTz6vn-Ah6HXYAw?e=i9ldNe)

### **Citation**

---