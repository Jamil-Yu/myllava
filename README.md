# Myllava

## 环境配置
```
git clone https://github.com/Jamil-Yu/myllava.git
cd myllava
conda create -n myllava python=3.10 -y
conda activate myllava
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
pip install protobuf
```

## 使用方法

在`myllava/llava/model/language_model/llava_llama.py`的generate函数中查看修改的部分，同时可以选择一些模式（本报告使用了`mode = "look at the image tokens"`与`mode = "mask according to the distance"`）

可以在`my_QA/answer/answer.json`中修改提问的图片与prompt，否则为默认图片。运行推理代码：
```
python llava/eval/model_vqa.py \    
    --model-path liuhaotian/llava-v1.5-7b \ 
    --question-file my_QA/question/question.jsonl \
    --image-folder my_QA/image \   
    --answers-file my_QA/answer/answer.json
```

