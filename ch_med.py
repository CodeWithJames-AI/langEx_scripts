import langextract as lx
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Text with a Chinese medicine mention
input_text = "【处方】金银花100g，连翘100g，桔梗60g，薄荷60g，淡豆豉50g，淡竹叶40g，牛蒡子60g，荆芥40g，芦根100g，甘草40g。"

# Define the prompt description
prompt_description = "Extract Chinese medicine names and their quantities from the text."
# Define example data with entities in order of appearance
examples = [
    lx.data.ExampleData(
        text="金银花100g，连翘100g，桔梗60g，薄荷60g。",
        extractions=[
            lx.data.Extraction(extraction_class="medicine_name", extraction_text="金银花"),
            lx.data.Extraction(extraction_class="quantity", extraction_text="100g"),
            lx.data.Extraction(extraction_class="medicine_name", extraction_text="连翘"),
            lx.data.Extraction(extraction_class="quantity", extraction_text="100g"),
            lx.data.Extraction(extraction_class="medicine_name", extraction_text="桔梗"),
            lx.data.Extraction(extraction_class="quantity", extraction_text="60g"),
            lx.data.Extraction(extraction_class="medicine_name", extraction_text="薄荷"),
            lx.data.Extraction(extraction_class="quantity", extraction_text="60g"),
        ]
    )
]

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt_description,
    examples=examples,
    model_id="gpt-4o",
    api_key=os.getenv("LANGEXTRACT_API_KEY"),  # Load from environment variable
    model_url=os.getenv("LANGEXTRACT_MODEL_URL", "https://api.laozhang.ai/v1"),  # Load from environment variable with fallback
    fence_output=True,
    use_schema_constraints=True,
)

# Display entities with positions
print(f"Input: {input_text}\n")
print("Extracted entities:")

for entity in result.extractions:
    position_info = ""
    if entity.char_interval:
        start, end = entity.char_interval.start_pos, entity.char_interval.end_pos
        position_info = f" (pos: {start}-{end})"
    print(f"• {entity.extraction_class.capitalize()}: {entity.extraction_text}{position_info}")

# Save and visualize the results
lx.io.save_annotated_documents([result], output_name="chinese_medicine_extraction.jsonl", output_dir=".")

# Generate the interactive visualization
html_content = lx.visualize("chinese_medicine_extraction.jsonl")
with open("chinese_medicine_extraction.html", "w", encoding="utf-8") as f:
    f.write(html_content)

print("Interactive visualization saved to chinese_medicine_extraction.html")