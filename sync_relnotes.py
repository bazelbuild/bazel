import re
import subprocess
import os
import time
import json
import google.generativeai as genai

def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None, None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')

def get_all_doc_paths():
    try:
        out = subprocess.check_output(
            ['git', '-C', 'bazel_src', 'ls-files', 'site/en/**/*.md'],
            text=True
        )
        return [p for p in out.split('\n') if p and '/versions/' not in p and '/archive/' not in p]
    except Exception as e:
        print(f"Error getting file list: {e}")
        return []

def find_best_docs_with_gemini(model, commit_subject, relnote, all_paths):
    paths_str = "\n".join(all_paths)
    prompt = f"""
    You are an expert Bazel engineer.
    Commit: {commit_subject}
    Note: {relnote}

    TASK: Find the MOST relevant document for this change.
    Return ONLY a JSON string of the path. Example: "site/en/ref.md"
    FILES:
    {paths_str}
    """
    try:
        time.sleep(1)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        match = re.search(r'site/en/[\w./-]+\.md', response.text.strip())
        if not match: return set()
        
        devsite = match.group(0)
        mintlify = devsite.replace("site/en/", "docs/").replace(".md", ".mdx")
        
        final = set()
        for p in [devsite, mintlify]:
            if os.path.exists(os.path.join('bazel_src', p)): final.add(p)
        return final
    except: return set()

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
    """PHASE 2: Rewrites or Replaces documentation based on the diff."""
    
    # 1. Generate the concise update text
    prompt_text = f"""
    Draft a concise documentation update (max 4 lines).
    Commit: {commit_subject}
    Note: {relnote_text}
    Diff: {commit_diff[:3000]}
    
    Instruction: If the diff shows a feature/flag being removed and replaced, 
    write the update to reflect the NEW state. Return ONLY raw markdown.
    """
    try:
        time.sleep(2)
        resp = model.generate_content(prompt_text, generation_config={"temperature": 0.0})
        new_text = resp.text.strip()
        new_text = re.sub(r'^```(?:markdown|mdx)?\s*|\s*```$', '', new_text, flags=re.IGNORECASE).strip()
    except Exception as e:
        print(f"  ❌ Error drafting text: {e}")
        return

    # 2. Surgical Placement / Replacement
    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        numbered_content = "".join([f"{i+1}: {line}" for i, line in enumerate(original_lines)])
        
        prompt_placement = f"""
        TASK: Update the document to reflect the code change.
        
        CODE CHANGE (Diff):
        {commit_diff[:2000]}
        
        NEW TEXT TO ADD:
        {new_text}
        
        DOCUMENT CONTENT:
        {numbered_content[:8000]}

        CRITICAL INSTRUCTIONS:
        1. If the Diff shows a feature is REMOVED or REPLACED, identify the line numbers in the document 
           describing the old feature and use "replace".
        2. If this is a brand new addition, use "insert_after".
        3. Return ONLY JSON: 
           {{"action": "replace", "start_line": 10, "end_line": 12}} 
           OR 
           {{"action": "insert_after", "line_number": 45}}
        """

        try:
            time.sleep(1)
            resp = model.generate_content(prompt_placement, generation_config={"temperature": 0.0})
            json_match = re.search(r'\{.*\}', resp.text.strip(), re.DOTALL)
            if not json_match: continue
            update = json.loads(json_match.group(0))

            # Apply MDX escaping
            file_text = new_text
            if doc_path.endswith('.mdx'):
                file_text = file_text.replace('{', '\\{').replace('}', '\\}')

            if update.get('action') == "replace":
                start = int(update['start_line']) - 1
                end = int(update['end_line'])
                if 0 <= start < len(original_lines):
                    print(f"  🔄 Replacing lines {start+1} to {end} in {doc_path}")
                    original_lines[start:end] = [file_text + "\n"]
            else:
                line_idx = int(update.get('line_number', 0)) - 1
                if 0 <= line_idx < len(original_lines):
                    print(f"  ✅ Inserting after line {line_idx+1} in {doc_path}")
                    original_lines.insert(line_idx + 1, "\n" + file_text + "\n")

            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(original_lines)
        except Exception as e:
            print(f"  ❌ Error in placement for {doc_path}: {e}")

def run_rulebook():
    flash_model, pro_model = setup_gemini()
    if not flash_model: return

    if not os.path.exists('weekly_notes.txt'): return
    all_doc_paths = get_all_doc_paths()

    with open('weekly_notes.txt', 'r', encoding='utf-8') as f:
        commits = f.read().split('COMMIT_DELIMITER\n')[1:]

    for block in commits:
        lines = block.strip().split('\n')
        if len(lines) < 3: continue
        chash, subject, body = lines[0], lines[1], '\n'.join(lines[2:])

        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match: continue
        
        # Robust Filtering for "None.", "N/A", etc.
        note = match.group(1).strip()
        clean_note = re.sub(r'[.*`#\s]+$', '', note).lower()
        if clean_note in ['none', 'n/a', 'no', '']: continue

        print(f"\n🚀 Processing {chash[:7]}: {subject[:50]}...")
        target_docs = find_best_docs_with_gemini(flash_model, subject, note, all_doc_paths)

        if target_docs:
            try:
                diff = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '--format=', chash], text=True).strip()
                rewrite_docs_with_gemini(pro_model, subject, note, diff, target_docs)
            except Exception as e:
                print(f"  ⚠️ Diff Error: {e}")

if __name__ == "__main__":
    run_rulebook()
