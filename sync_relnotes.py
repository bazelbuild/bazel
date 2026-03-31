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
    # Changed to gemini-1.5 versions as 2.5 is not currently a release version
    return genai.GenerativeModel('gemini-1.5-flash'), genai.GenerativeModel('gemini-1.5-pro')

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
    """PHASE 1: Asks Gemini to pick exactly 1 .md and 1 .mdx file."""
    paths_str = "\n".join(all_paths)
    prompt = f"""
    You are an expert Bazel engineer. 
    Commit: {commit_subject}
    Note: {relnote}

    Select EXACTLY ONE 'site/en/*.md' file and EXACTLY ONE 'docs/*.mdx' file that should be updated.
    Return ONLY a JSON array of the two paths. If no match, return [].

    --- FILES ---
    {paths_str}
    """
    try:
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        raw_json = re.sub(r'^```(?:json)?\s*|\s*```$', '', response.text.strip(), flags=re.IGNORECASE)
        return set(json.loads(raw_json))
    except:
        return set()

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
    """PHASE 2: Surgically updates the file with minimal (4-5 lines) changes."""

    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)
        if not os.path.exists(full_path): continue

        with open(full_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        # Provide the file with line numbers to Gemini for surgical accuracy
        numbered_content = "".join([f"{i+1}: {line}" for i, line in enumerate(original_lines)])

        prompt = f"""
        You are an expert technical writer. Update the documentation based on this code change.

        Commit Subject: {commit_subject}
        Release Note: {relnote_text}
        Code Diff: {commit_diff[:3000]}

        CRITICAL RULES:
        1. Make a MINIMAL change (exactly 4 to 5 lines).
        2. Do not rewrite the whole file.
        3. Identify the most relevant line number to insert or replace text.
        4. If it's an MDX file, escape bare braces ({{ -> \\{{).
        5. Output your response in EXACTLY this JSON format:
        {{
            "action": "insert_after" or "replace",
            "line_number": 123,
            "new_text": "The 4-5 lines of documentation content..."
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
            new_text = update['new_text']
            
            # Perform surgical edit
            if update['action'] == "replace":
                original_lines[line_idx] = new_text + "\n"
            else: # insert_after
                original_lines.insert(line_idx + 1, "\n" + new_text + "\n")

            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(original_lines)

            print(f"  ✅ Surgically updated {doc_path} (Line {update['line_number']})")

        except Exception as e:
            print(f"  ❌ Gemini Error for {doc_path}: {e}")

def run_rulebook():
    models = setup_gemini()
    if not models: return
    flash_model, pro_model = models

    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path): return

    with open(log_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()

    all_doc_paths = get_all_doc_paths()
    commits = raw_data.split('COMMIT_DELIMITER\n')[1:]

    for commit_block in commits:
        lines = commit_block.strip().split('\n')
        if len(lines) < 3: continue

        commit_hash, commit_subject = lines[0].strip(), lines[1].strip()
        body = '\n'.join(lines[2:])

        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match: continue
        note = match.group(1).strip()

        # Skip trivial notes
        if re.sub(r'[*`]', '', note).lower().strip() in ['none', 'n/a', 'no']: continue

        print(f"\n✅ Processing: {commit_hash[:7]}")
        target_docs = find_best_docs_with_gemini(flash_model, commit_subject, note, all_doc_paths)

        if target_docs:
            try:
                diff = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '--format=', commit_hash], text=True).strip()
                rewrite_docs_with_gemini(pro_model, commit_subject, note, diff, target_docs)
            except Exception as e:
                print(f"  ⚠️ Error fetching diff: {e}")

if __name__ == "__main__":
    run_rulebook()
