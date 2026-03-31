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
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')

def get_all_doc_paths():
    """Gets a list of all current .md and .mdx files in the repo."""
    try:
        out = subprocess.check_output(
            ['git', '-C', 'bazel_src', 'ls-files', 'site/en/**/*.md', 'docs/**/*.mdx'], 
            text=True
        )
        paths = [p for p in out.split('\n') if p and '/versions/' not in p and '/archive/' not in p]
        return paths
    except Exception as e:
        print(f"Error getting file list: {e}")
        return []

def find_best_docs_with_gemini(model, commit_subject, relnote, all_paths):
    """PHASE 1: Picks exactly one .md and its corresponding .mdx file."""
    paths_str = "\n".join(all_paths)
    prompt = f"""
    You are an expert Bazel engineer. 
    Commit: {commit_subject}
    Release Note: {relnote}

    TASK:
    Find the MOST relevant documentation file for this change.
    You MUST return EXACTLY TWO files:
    1. One DevSite file (starts with 'site/en/' and ends in '.md').
    2. One Mintlify file (starts with 'docs/' and ends in '.mdx').
    These two files MUST cover the same topic.

    Return ONLY a JSON array of strings. Example: ["site/en/ref.md", "docs/ref.mdx"]
    If no relevant docs exist, return [].

    --- FILES ---
    {paths_str}
    """
    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        raw_json = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.text.strip(), flags=re.IGNORECASE)
        paths = json.loads(raw_json)
        return set(paths)
    except:
        return set()

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
    """PHASE 2: Surgically updates the file with a strict 4-line limit."""

    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)
        if not os.path.exists(full_path): continue

        with open(full_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        numbered_content = "".join([f"{i+1}: {line}" for i, line in enumerate(original_lines)])

        prompt = f"""
        You are a technical writer. Update this Bazel doc with a MINIMAL note.
        
        Commit: {commit_subject}
        Note: {relnote_text}
        Diff: {commit_diff[:2000]}

        CRITICAL RULES:
        1. Add or Replace a MAXIMUM of 4 lines. DO NOT write paragraphs.
        2. Be extremely concise. Use one bullet point or two short sentences.
        3. If this is an MDX file, escape braces ({{ -> \\{{).
        4. Return ONLY JSON:
        {{
            "action": "insert_after" or "replace",
            "line_number": 123,
            "new_text": "Your 1-4 lines of text here"
        }}

        --- DOCUMENT ({doc_path}) ---
        {numbered_content}
        """

        try:
            time.sleep(2) 
            response = model.generate_content(prompt, generation_config={"temperature": 0.0})
            raw_json = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.text.strip(), flags=re.IGNORECASE)
            update = json.loads(raw_json)

            line_idx = int(update['line_number']) - 1
            new_text = update['new_text'].strip()
            
            # Hard enforcement of 4-line limit in Python
            lines_to_add = new_text.split('\n')
            if len(lines_to_add) > 4:
                print(f"  ⚠️ AI tried to add {len(lines_to_add)} lines. Truncating to 4.")
                new_text = "\n".join(lines_to_add[:4])

            if update['action'] == "replace":
                original_lines[line_idx] = new_text + "\n"
            else:
                # insert_after
                original_lines.insert(line_idx + 1, "\n" + new_text + "\n")

            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(original_lines)

            print(f"  ✅ Surgical Update: {doc_path} ({len(lines_to_add)} lines added/changed)")

        except Exception as e:
            print(f"  ❌ Error for {doc_path}: {e}")

def run_rulebook():
    flash_model, pro_model = setup_gemini()
    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path): return

    all_doc_paths = get_all_doc_paths()
    
    with open(log_path, 'r', encoding='utf-8') as f:
        commits = f.read().split('COMMIT_DELIMITER\n')[1:]

    for commit_block in commits:
        lines = commit_block.strip().split('\n')
        if len(lines) < 3: continue

        commit_hash, commit_subject = lines[0].strip(), lines[1].strip()
        body = '\n'.join(lines[2:])
        
        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match: continue
        note = match.group(1).strip()
        if re.sub(r'[*`]', '', note).lower().strip() in ['none', 'n/a']: continue

        print(f"\n🚀 Processing: {commit_hash[:7]} - {commit_subject[:50]}...")
        
        # Phase 1: Force find the .md / .mdx PAIR
        target_docs = find_best_docs_with_gemini(flash_model, commit_subject, note, all_doc_paths)

        if target_docs:
            print(f"  🎯 Target Pair: {list(target_docs)}")
            try:
                diff = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '--format=', commit_hash], text=True).strip()
                # Phase 2: Perform minimal edit on BOTH files in the pair
                rewrite_docs_with_gemini(pro_model, commit_subject, note, diff, target_docs)
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
        else:
            print("  ⏭️ No matching documentation pair found.")

if __name__ == "__main__":
    run_rulebook()
