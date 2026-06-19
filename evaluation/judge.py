from openai import OpenAI
import time
import pandas as pd
import random
from tqdm import tqdm
# ==============================
# vLLM API 配置
# ==============================
client = OpenAI(
    api_key="EMPTY",
    base_url="http://localhost:8000/v1",
)

model_name = "Qwen/Qwen3-8B"
max_tokens = 1000


sys_prompt = """
You are a strict evaluator.

You will be given:
1. A list or single ground truth answers (multiple valid formats).
2. A predicted answer, which may contain extra text or explanations.

Your task is to determine whether the prediction is correct.

----------------------
Evaluation Rules:

1. The prediction is considered CORRECT if it contains ANY one of the ground truth answers as a substring, ignoring minor formatting differences such as:
   - punctuation (e.g., ":" vs ".")
   - extra spaces
   - capitalization

2. Ignore any irrelevant text in the prediction, including reasoning, explanations, or content inside <think> tags.

3. Focus only on whether the core answer content matches one of the ground truth answers.

4. The prediction is INCORRECT if:
   - none of the valid answers appear in the prediction
   - or the meaning is clearly different

----------------------

Output format (strict):
- Output only one word: "CORRECT" or "INCORRECT"
- Do NOT output anything else
"""


def query(user_prompt, temperature, max_tokens, max_retries=20):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=1,
                extra_body={
                    "top_k": 20,  # vLLM 支持
                    "chat_template_kwargs": {"enable_thinking": False},
                },
            )
            return response.choices[0].message.content

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"Error (attempt {attempt + 1}/{max_retries}): {e}, retrying...")
                time.sleep(2 ** attempt)
            else:
                print(f"Failed after {max_retries} attempts.")
                raise e


def process_excel(file_path, output_path):
    print(f"Reading {file_path}...")
    df = pd.read_excel(file_path)

    if 'answer' not in df.columns or 'prediction' not in df.columns:
        raise ValueError("The excel file must contain 'answer' and 'prediction' columns.")

    evaluations = []
    total_rows = len(df)

    for index, row in tqdm(df.iterrows(),total =total_rows ):
        gt_answer = str(row['answer'])
        prediction = str(row['prediction'])
        user_prompt = f"Ground Truth(s): {gt_answer}\nPrediction: {prediction}"
        try:
            result = query(user_prompt, temperature=0.0, max_tokens=max_tokens)
            evaluations.append(result.strip())
        except Exception as e:
            print(f"Failed at row {index + 1}: {e}")
            evaluations.append("ERROR")

    df['evaluation'] = evaluations
    df.to_excel(output_path, index=False)

    correct_count = sum(1 for e in evaluations if e == "CORRECT")
    accuracy = correct_count / total_rows if total_rows > 0 else 0

    print(f"Finished. Saved to {output_path}")
    print(f"Accuracy: {accuracy:.2%} ({correct_count}/{total_rows})")


if __name__ == "__main__":
    input_xlsx = "/path/to/your/ChartQA_TEST.xlsx"  # replace with your actual path
    output_xlsx = "result_ChartQA_TEST.xlsx"

    process_excel(input_xlsx, output_xlsx)