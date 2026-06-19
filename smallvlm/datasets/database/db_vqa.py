import json
import copy
from .db import register_dataset, ImageVQADataBase, ImageVQAParquetDataBase

def add_image_token(instruction, ref_text=None):
    if ref_text is None:
        ref_text = instruction
    n = sum(ord(c) * (i + 1) for i, c in enumerate(ref_text))
    instruction = "<image>\n" + instruction if n % 2 else instruction + "\n<image>"

    return instruction

def convert_type(conversation):
    if "value" in conversation[0].keys() and conversation[0]['value'] != None:
        for msg in conversation:
            if msg["from"] != "system" and len(msg["value"])==0 :
                return None
        messages = [{"from": msg["from"], "value": msg["value"]} for msg in conversation if len(msg["value"])>0]
    else:
        for msg in conversation:
            if len(msg["content"])==0:
                return None
        messages = [{"from": "human" if msg["role"] == "user" else "gpt", "value": msg["content"]} for msg in conversation]
    return messages

def remove_leading_newlines_from_assistant(conversation):
    # 遍历 "conversation" 键中的每个对话
    for conv in conversation:
        if conv["role"] == "assistant":
            # 获取 "text" 内容
            for content in conv["content"]:
                if content["type"] == "text":
                    # 去掉文本开头的换行符
                    content["text"] = content["text"].lstrip('\n ')
    return conversation


@register_dataset(naming=True)
class LlavaDataset(ImageVQADataBase):
    data_type = "llava"

    def get_conv(self, record):
        conversation = record["conversations"]
        if "mask_rle" in record:
            conversation = conversation.copy()
            conversation[0] = conversation[0].copy()
            conversation[0]["value"] = "<image>\n<image>\nThe first image is the full scene.The second image is a cropped region from the first image.Describe the cropped region in detail."
        images = record.get("images", [])
        new_conversation = []
        image_idx = 0
        for msg in conversation:
            assert "from" in msg and "value" in msg, f"Invalid message format: {msg}"
            content = []
            if msg["from"] == "human":
                texts = msg["value"].split("<image>")
                for i, text in enumerate(texts):
                    text = text.strip()
                    if text:
                        content.append({"type": "text", "text": text})
                    if i < len(texts) - 1 and len(images) != 0:
                        content.append({"type": "image", "image": images[image_idx]})
                        image_idx += 1
                
                new_conversation.append({"role":"user", "content": content})

            elif msg["from"] == "gpt":
                new_conversation.append({"role": "assistant", "content": [{"type": "text", "text": msg["value"]}]})
                
        new_conversation = remove_leading_newlines_from_assistant(new_conversation)
        if conversation[0]['from'] == 'system':
            for conv in new_conversation:
                if conv['role'] == 'user':
                    for item in conv['content']:
                        if item['type'] == 'text':
                            item['text'] = item['text']+ "\n"+conversation[0]["value"]

        meta = {"conversation": new_conversation}
        if "id" in record:
            meta["id"] = record["id"]
        return meta


@register_dataset(naming=True)
class LlavaParquetDataset(ImageVQAParquetDataBase):
    data_type = "parquet"

    def get_conv(self, record):
        conversation = copy.deepcopy(record["conversations"])
        if not isinstance(conversation, list):
            if isinstance(conversation, str):
                conversation = json.loads(conversation)
            else:
                conversation = list(conversation)

        conversation = convert_type(conversation)
        images = record.get("images", [])
        
        new_conversation = []
        image_idx = 0
        for msg in conversation:

            assert "from" in msg and "value" in msg, f"Invalid message format: {msg}"
            content = []
            
            if msg["from"] == "human":
                texts = msg["value"].split("<image>")
                for i, text in enumerate(texts):
                    text = text.strip()
                    if text:
                        content.append({"type": "text", "text": text})
                    if i < len(texts) - 1 and len(images) != 0:

                        content.append({"type": "image", "image": images[image_idx]})
                        image_idx += 1

                new_conversation.append({"role": "user", "content": content})

            elif msg["from"] == "gpt":
                new_conversation.append({"role": "assistant", "content": [{"type": "text", "text": msg["value"]}]})

        meta = {"conversation": new_conversation}
        if "id" in record:
            meta["id"] = record["id"]
        return meta

    

    



