import torch
import tqdm
import json
from transformers import AutoModelForCausalLM, AutoTokenizer


class Tools:
    @staticmethod
    def load_jsonl(path):
        with open(path, 'r') as f:
            return [json.loads(line) for line in f.readlines()]
    
    @staticmethod
    def dump_jsonl(obj, path):
        with open(path, 'w') as f:
            for line in obj:
                f.write(json.dumps(line) + '\n')


class CodeGen:
    def __init__(self, model_name, batch_size):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.add_special_tokens({'pad_token': self.tokenizer.eos_token})
        self.model.cuda()
        self.batch_size = batch_size
        print('done loading model')
    
    def _get_batchs(self, prompts, batch_size):
        batches = []
        for i in range(0, len(prompts), batch_size):
            batches.append(prompts[i:i+batch_size])
        return batches

    def _generate_batch(self, prompt_batch, max_new_tokens=100):
        prompts = self.tokenizer(prompt_batch, return_tensors='pt', padding=True, truncation=True)

        with torch.no_grad():
            gen_tokens = self.model.generate(
                input_ids = prompts['input_ids'].cuda(),
                attention_mask = prompts['attention_mask'].cuda(),
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
        gen_text = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        for i in range(len(gen_text)):  # remove the prompt
            gen_text[i] = gen_text[i][len(prompt_batch[i]):]
        return gen_text

    def batch_generate(self, file):
        print(f'generating from {file}')
        lines = Tools.load_jsonl(file)
        # have a new line at the end
        prompts = [f"{line['prompt']}\n" for line in lines]
        batches = self._get_batchs(prompts, self.batch_size)
        gen_text = []
        for batch in tqdm.tqdm(batches):
            gen_text.extend(self._generate_batch(batch))
        print(f'generated {len(gen_text)} samples')
        assert len(gen_text) == len(prompts)
        new_lines = []
        for line, gen in zip(lines, gen_text):
            new_lines.append({
                'prompt': line['prompt'],
                'metadata': line['metadata'],
                'choices': [{'text': gen}]
            })
        Tools.dump_jsonl(new_lines, file.replace('.jsonl', f'_{self.model_name.split("/")[-1]}.jsonl'))


if __name__ == '__main__':
    file_path = 'datasets/line_level_completion_1k_context_codegen.test.jsonl'
    tiny_codegen = 'Salesforce/codegen-350M-mono'

    cg = CodeGen(tiny_codegen, batch_size=8)
    cg.batch_generate(file_path)
