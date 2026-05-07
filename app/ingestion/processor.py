"""
Process data from knowledgebase to clean text for embeddings.
"""
# import pymupdf.layout  # to_json
import pymupdf4llm
import os
import re
from PIL import Image
from groq import Groq
import base64
from pathlib import Path
from typing import List
from tiktoken import get_encoding
import imagehash
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch")
from dotenv import load_dotenv
import json
import fitz
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=groq_api_key)

enc = get_encoding("cl100k_base")

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    full_md = []

    for page_num in range(len(doc)):

        md = pymupdf4llm.to_markdown(
            doc=doc,
            pages=[page_num],
            write_images=True,
            image_path="images",
            image_format="png",
            dpi=300,
            extract_words=False
        )

        full_md.append(f"\n\n<<<PAGE:{page_num+1}>>>\n\n")
        full_md.append(md)

    # pathlib.Path("output.md").write_text("".join(full_md), encoding="utf-8")

    return "".join(full_md)

def parse_model_json(raw_text):
    cleaned = re.sub(r"```json|```", "", raw_text).strip()
    return json.loads(cleaned)

def img_caption(img_path):

    with open(img_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": """Analyze this image.
                    Return the output as string.
                    Rules (MANDATORY):
                    - If the image is primarily a logo or decorative branding with no informational value, **ALWAYS** return "logo detected" as plain text ONLY.
                    - Otherwise for non logo images return a structured semantic representation in markdown format and include  all relevant sections that apply from IMAGE TYPE, ENTITIES, RELATIONSHIPS, AXES, TABLES, COMPLETE SUMMARY, VISIBLE TEXT. Use the section lables as headings.
                    - Do not invent details.
                    """},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"}
                    }
                ]
            }
        ],
        temperature=0,
        max_completion_tokens=1024
    )

    return response.choices[0].message.content

def ocr_caption_images(text, logo_threshold=0.7, delete_images=True):
    logo_hash_counts = {}
    logo_hashes = set()
    img_hashes = {}
    # Step 1: Compute hashes for all images
    image_paths = re.findall(r'!\[.*?\]\((.*?)\)', text)
    valid_imgs = 0
    for img_path in image_paths:
        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                img_obj = img.convert("RGB")
                h = str(imagehash.phash(img_obj))
            img_hashes[img_path] = h
            logo_hash_counts[h] = logo_hash_counts.get(h, 0) + 1
            valid_imgs += 1
    if valid_imgs>2:
        logo_hashes = {h for h, count in logo_hash_counts.items() if count /valid_imgs >= logo_threshold}
    # Step 2: Remove logos
    for img_path in image_paths:
        remove_logo = False
        img_hash = None

        # Find matching metadata
        if os.path.exists(img_path):
            img_hash = img_hashes.get(img_path)

        if img_hash in logo_hashes:
            remove_logo = True
        
        if remove_logo:
            # Remove references in text
            pattern = rf'!\[[^\]]*\]\(\s*{re.escape(img_path)}\s*\)'
            text = re.sub(pattern, '', text)

            # Delete the file if requested
            if delete_images and os.path.exists(img_path):
                os.remove(img_path)
        

    # Step 3: OCR & caption non-logo images
    image_tags = re.findall(r'!\[.*?\]\(.*?\)', text)

    for  img_tag in image_tags:
        img_path_match = re.search(r'!\[.*?\]\((.*?)\)', img_tag)
        if not img_path_match:
            continue
        img_path = img_path_match.group(1)
        # Only process if image still exists and is non-logo
        if os.path.exists(img_path):
            caption_for_img = img_caption(img_path)
            # fallback logo detection
            if caption_for_img == 'logo detected':   
                # Remove logo reference
                pattern = rf'!\[[^\]]*\]\(\s*{re.escape(img_path)}\s*\)'
                text = re.sub(pattern, '', text)
                if delete_images:
                    os.remove(img_path)
                continue
                
            # Replace tag in text
            replacement = f"""

            
### IMAGE_BLOCK_START
Image
**Source:** {img_path}

**STRUCTURED_IMAGE_DATA:** {caption_for_img}

IMAGE_BLOCK_END
"""
            text = re.sub(re.escape(img_tag), replacement, text, count=1)
    return text

def generate_doc_id(file_path):
    name = Path(file_path).stem  # remove extension
    name = name.lower()
    name = re.sub(r'[^a-z0-9]+', '_', name)
    name = name.strip('_')
    return name

def extract_blocks(text, doc_id, start_index=0):

    blocks = []
    idx = start_index
    current_page = None

    image_pattern = r"### IMAGE_BLOCK_START.*?IMAGE_BLOCK_END"
    image_blocks = re.findall(image_pattern, text, re.DOTALL)

    placeholders = {}

    for i, block in enumerate(image_blocks):
        key = f"__IMG_BLOCK_{i}__"
        placeholders[key] = block
        text = text.replace(block, key)

    raw_blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

    i = 0

    while i < len(raw_blocks):

        raw = raw_blocks[i]

        # Detect page markers
        if raw.startswith("<<<PAGE:"):
            page_match = re.search(r"\d+", raw)
            if page_match:
                current_page = int(page_match.group())
            i += 1
            continue

        if raw.startswith("__IMG_BLOCK_"):

            block_type = "image"
            raw = placeholders[raw]

            content = raw.replace("### IMAGE_BLOCK_START","").replace("IMAGE_BLOCK_END","").strip()
            metadata = {}

        elif raw.startswith("#"):

            block_type = "heading"
            content = raw
            metadata = {}

        elif raw.startswith("|"):

            block_type = "table"

            table_lines = []
            while i < len(raw_blocks) and raw_blocks[i].startswith("|"):
                table_lines.append(raw_blocks[i])
                i += 1

            content = "\n".join(table_lines)
            metadata = {}
            i -= 1

        else:

            block_type = "paragraph"
            content = raw
            metadata = {}

        block = {
            "block_id": f"{doc_id}_block_{idx}",
            "doc_id": doc_id,
            "page": current_page, 
            "type": block_type,
            "content": content,
            "metadata": metadata
        }

        blocks.append(block)

        idx += 1
        i += 1

    return blocks

def count_tokens(text: str) -> int:
    """Return number of tokens in a string."""
    return len(enc.encode(text))

def handle_table_block(table_text: str, max_tokens: int, has_header: bool = None, page_number: int = None):
    """
    Split table into chunks if larger than max_tokens.
    Repeat header if known, otherwise add TABLE CONTINUED marker.
    """
    # TODO: add borderless table detection
    lines = table_text.strip().split("\n")
    
    # Detect header if unknown
    if has_header is None:
        has_header = len(lines) > 1 and all('-' in c for c in lines[1].split('|') if c.strip())
    
    if has_header:
        header = lines[0]
        start_idx = 1
    else:
        header = None
        start_idx = 0
    
    chunks = []
    current_chunk = []
    current_tokens = 0
    
    if header:
        current_chunk.append(header)
        current_tokens += count_tokens(header)
    
    for line in lines[start_idx:]:
        line_tokens = count_tokens(line)
        if current_tokens + line_tokens > max_tokens:
            # finalize current chunk
            chunks.append("\n".join(current_chunk))
            # start new chunk
            current_chunk = []
            current_tokens = 0

            if page_number is not None:
                current_chunk.append(f"TABLE CONTINUED (from page {page_number})")
                current_tokens += count_tokens(current_chunk[-1])
            if header:
                current_chunk.append(header)
                current_tokens += count_tokens(header)
                 
        current_chunk.append(line)
        current_tokens += line_tokens
    
    if current_chunk:
        chunks.append("\n".join(current_chunk))
    
    return chunks


def assemble_context_segments(blocks: List[dict], max_tokens: int = 800, overlap: int = 120):
    
    segments = []
    current_tokens = 0
    current_content = []
    current_blocks = []
    
    segment_idx = 0
    pending_heading = None

    for block in blocks:
        block_text = block["content"]
        # block_tokens = count_tokens(block_text)

        # Attach heading to next content block
        if block["type"] == "heading":
            pending_heading = block_text
            continue

        if block["type"] == "table":
        # Split table into chunks if it exceeds max_tokens
            table_chunks = handle_table_block(
                table_text=block["content"],
                max_tokens=max_tokens,
                has_header=None,       # detect header automatically
                page_number=block["page"]
            )
        else:
            # imgs and paragraphs
            table_chunks = [block["content"]]

        for chunk_text in table_chunks:
            chunk_tokens = count_tokens(chunk_text)

            # Prepend heading if pending
            if pending_heading:
                chunk_text = pending_heading + "\n" + chunk_text
                chunk_tokens += count_tokens(pending_heading)
                pending_heading = None

            # If adding this chunk exceeds max_tokens, finalize current segment
            if current_tokens + chunk_tokens > max_tokens and current_content:
                segment_text = "\n\n".join(current_content)
                pages = sorted({b["page"] for b in current_blocks})
                segment = {
                    "segment_id": f"{block['doc_id']}_segment_{segment_idx}",
                    "doc_id": block["doc_id"],
                    "blocks": current_blocks.copy(),
                    "content": segment_text, 
                    "pages": pages,
                    "page_range": f"{min(pages)}-{max(pages)}" if len(pages) > 1 else str(pages[0])
                }
                segments.append(segment)
                segment_idx += 1

                # Reset for next chunk
                if block["type"] != "table" and overlap > 0:
                    # Keep last `overlap` tokens (approx)
                    tokens_accum = 0
                    overlap_content = []
                    overlap_blocks = []
                    for i in reversed(range(len(current_content))):
                        part_text = current_content[i]
                        part_tokens = count_tokens(part_text)
                        if tokens_accum + part_tokens <= overlap:
                            overlap_content.insert(0, part_text)
                            overlap_blocks.insert(0, current_blocks[i])
                            tokens_accum += part_tokens
                        else:
                            break
                    current_content = overlap_content
                    current_blocks = overlap_blocks
                    current_tokens = tokens_accum
                else:
                    current_content = []
                    current_blocks = []
                    current_tokens = 0

            # Add chunk to current segment
            current_content.append(chunk_text)
            current_blocks.append({
                "block_id": block["block_id"],
                "page": block["page"],
                "type": block["type"]
            })
            current_tokens += chunk_tokens  # <- ensure this is after reset logic


    # Add remaining content as last segment
    if current_content:
        segment_text = "\n\n".join(current_content)
        pages = sorted({b["page"] for b in current_blocks})
        segment = {
            "segment_id": f"{blocks[-1]['doc_id']}_segment_{segment_idx}",
            "doc_id": blocks[-1]["doc_id"],
            "blocks": current_blocks.copy(),
            "content": segment_text,
            # "doc_title": block['doc_id'],  
            "pages": pages,
            "page_range": f"{min(pages)}-{max(pages)}" if len(pages) > 1 else str(pages[0])
            
        }
        segments.append(segment)

    return segments 


def filter_chunks(text, file_path):
    doc_metadata = {} 

    doc_id = generate_doc_id(file_path)
    doc_metadata['doc_id'] = doc_id

    all_blocks = []
    all_blocks = extract_blocks(text, doc_id)
    
    segments = assemble_context_segments(all_blocks, max_tokens= 800, overlap=80)

    return segments
        

def process_text(filename):
    """ Extract text from a PDF file and append necessary metadata for embeddings."""
    
    text = extract_text_from_pdf(filename)
    text = ocr_caption_images(text)
    segments = filter_chunks(text,filename)
    return segments


