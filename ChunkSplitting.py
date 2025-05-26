import pandas as pd
from bs4 import BeautifulSoup
import tiktoken

# Config
INPUT_CSV = "ketto_faq_data.csv"
OUTPUT_CSV = "ketto_kb_chunks.csv"
MAX_TOKENS = 300
ENCODING = tiktoken.get_encoding("cl100k_base")

print("🔍 Loading CSV...")
df = pd.read_csv(INPUT_CSV)

# Sanity check
print("🧾 Columns found:", df.columns.tolist())
print("🔍 Previewing data:")
print(df[['question', 'answer']].head())

# Clean missing data
df['question'] = df['question'].fillna('').astype(str)
df['answer'] = df['answer'].fillna('').astype(str)

# Check if content is missing
print("\n📊 Missing values count:")
print(df[['question', 'answer']].isnull().sum())

# Use plain text fallback chunker
def smart_semantic_chunks(text):
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    chunks = []
    current_chunk = []
    current_token_count = 0

    for para in paragraphs:
        token_count = len(ENCODING.encode(para))
        if current_token_count + token_count > MAX_TOKENS:
            if current_chunk:
                chunks.append("\n\n".join(current_chunk).strip())
            current_chunk = [para]
            current_token_count = token_count
        else:
            current_chunk.append(para)
            current_token_count += token_count

    if current_chunk:
        chunks.append("\n\n".join(current_chunk).strip())

    return chunks

# Process rows
records = []
print("\n🔄 Processing rows...")
for idx, row in df.iterrows():
    full_text = f"**Q: {row['question']}**\n\n{row['answer']}".strip()
    if not full_text:
        print(f"⚠️ Skipping empty row {idx}")
        continue

    try:
        chunked = smart_semantic_chunks(full_text)
        print(f"✅ Row {idx}: {len(chunked)} chunks created.")

        for i, chunk in enumerate(chunked):
            records.append({
                "chunk_id": f"{row['page_url']}_{i}",
                "page_url": row["page_url"],
                "page_title": row["page_title"],
                "question": row["question"],
                "answer": row["answer"],
                "chunk_index": i,
                "content_chunk": chunk,
                "token_count": len(ENCODING.encode(chunk))
            })

    except Exception as e:
        print(f"❌ Error processing row {idx}: {e}")

# Save to CSV
if records:
    chunked_df = pd.DataFrame(records)
    chunked_df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
    print(f"\n✅ Saved {len(chunked_df)} chunks to {OUTPUT_CSV}")
else:
    print("\n No chunks were created. Check content or chunking logic.")
