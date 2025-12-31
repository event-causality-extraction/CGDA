# 将json数据转化为fastie适用的jsonl数据
import json

def find_entity_offsets(sentence, entity_text):
    start = sentence.find(entity_text)
    if start == -1:
        print(f"Warning: Entity '{entity_text}' not found in sentence: {sentence}")
        return None
    end = start + len(entity_text)
    return start, end

def convert_format1_to_format2(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    with open(output_file, 'w', encoding='utf-8') as out_f:
        for item in data:
            sentence = item['sentence']
            cause = item['cause']
            effect = item['effect']

            entities = []

            # 处理 cause
            offsets = find_entity_offsets(sentence, cause)
            if offsets is not None:
                start, end = offsets
                entities.append({
                    "entity": cause,
                    "start_offset": start,
                    "end_offset": end,
                    "label": "cause"
                })

            # 处理 effect
            offsets = find_entity_offsets(sentence, effect)
            if offsets is not None:
                start, end = offsets
                entities.append({
                    "entity": effect,
                    "start_offset": start,
                    "end_offset": end,
                    "label": "effect"
                })

            output_item = {
                "text": sentence,
                "entities": entities
            }

            out_f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

# 使用示例
if __name__ == "__main__":
    convert_format1_to_format2('/data01/hujunjie/CGDA_REBUTTAL/baseline/eda/fcr/doc/train_full_eda.json', '/data01/hujunjie/CGDA_REBUTTAL/val_model/GlobalPointer/FastIE/examples/named_entity_recognition/datasets/cgda/eda/fcr/train.json')