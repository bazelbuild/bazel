import re
import subprocess
import os
import time
import json
import google.generativeai as genai

def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key: return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')

def get_all_doc_paths():
    try:
        out = subprocess.check_output(['git', '-C', 'bazel_src', 'ls-files', 'site/en/**/*.md', 'docs/**/*.mdx'], text=True)
        return [p for p in out.split('\n') if p and '/versions/' not in p and '/archive/' not in p]
    except: return []

def find_best_docs_with_gemini(model, commit_subject, relnote, all_paths):
    paths_str = "\n".join(all_paths)
    prompt = f"""
    You are an expert Bazel engineer.
    Find the MOST relevant file in 'site/en/' (.md) and its EXACT counterpart in 'docs/' (.mdx).
    Return ONLY a JSON array of the two paths.
    --- FILES ---
    {paths_str}
    """
    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        raw_json = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.text.strip(), flags=re.IGNORECASE)
        return set(json.loads(raw_json))
    except: return set()

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)
        if not os.path.exists(full_path): continue
        is_mdx = doc_path.endswith('.mdx')

        with open(full_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        numbered_content = "".join([f"{i+1}: {line}" for i, line in enumerate(original_lines)])

        prompt = f"""
        Update this {'Mintlify MDX' if is_mdx else 'DevSite MD'} doc. 
        Limit: 4 lines. {'If MDX, escape braces {{ }}.' if is_mdx else ''}
        
        Return ONLY JSON:
        {{
            "action": "insert_after",
            "line_number": 10,
            "new_text": "text"
        }}
        --- DOC ({doc_path}) ---
        {numbered_content}
        """

        try:
            time.sleep(2)
            response = model.generate_content(prompt, generation_config={"temperature": 0.0})
            raw_json = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.text.strip(), flags=re.IGNORECASE)
            update = json.loads(raw_json)

            line_idx = int(update['line_number']) - 1
            new_text = update['new_text'].strip()

            if is_mdx:
                new_text = new_text.replace("{", "\\{").replace("}", "\\}")

            # Apply change
            if update['action'] == "replace":
                original_lines[line_idx] = new_text + "\n"
            else:
                original_lines.insert(line_idx + 1, "\n" + new_text + "\n")

            # Write file to disk
            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(original_lines)

            # --- THE FIX: AUTOMATICALLY STAGE THE FILE FOR GIT ---
            subprocess.run(['git', '-C', 'bazel_src', 'add', doc_path])
            
            print(f"  ✅ Updated & Staged: {doc_path}")
            print(f"     Preview: {new_text[:60]}...")

        except Exception as e:
            print(f"  ❌ Error for {doc_path}: {e}")

def run_rulebook():
    flash_model, pro_model = setup_gemini()
    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path): return
    all_paths = get_all_doc_paths()
    
    with open(log_path, 'r', encoding='utf-8') as f:
        commits = f.read().split('COMMIT_DELIMITER\n')[1:]

    for block in commits:
        lines = block.strip().split('\n')
        if len(lines) < 3: continue
        c_hash, c_subj = lines[0].strip(), lines[1].strip()
        
        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', '\n'.join(lines[2:]), re.IGNORECASE)
        if not match: continue
        note = match.group(1).strip()
        if note.lower() in ['none', 'n/a']: continue

        print(f"\n🚀 Processing: {c_hash[:7]}...")
        target_docs = find_best_docs_with_gemini(flash_model, c_subj, note, all_paths)

        if target_docs:
            diff = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '--format=', c_hash], text=True).strip()
            rewrite_docs_with_gemini(pro_model, c_subj, note, diff, target_docs)

if __name__ == "__main__":
    run_rulebook()
