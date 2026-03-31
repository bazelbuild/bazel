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
    """Asks Gemini to pick exactly 1 .md and 1 .mdx file from the list."""
    paths_str = "\n".join(all_paths)
    prompt = f"""
    You are an expert Bazel engineer. A new commit was merged with this release note:
    Subject: {commit_subject}
    Release Note: {relnote}

    Find the MOST appropriate file to document this change.
    RULES:
    1. Select EXACTLY ONE DevSite file (site/en/*.md).
    2. Select EXACTLY ONE Mintlify file (docs/*.mdx).
    3. Return ONLY a JSON array. If no fit, return [].

    --- AVAILABLE FILES ---
    {paths_str}
    """
    try:
        time.sleep(1)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        raw_json = response.text.strip()
        raw_json = re.sub(r'^```(json)?\s*', '', raw_json, flags=re.IGNORECASE)
        raw_json = re.sub(r'\s*```$', '', raw_json).strip()
        selected_paths = json.loads(raw_json)
        return set(selected_paths)
    except Exception as e:
        print(f"  ⚠️ Gemini discovery error: {e}")
        return set()

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
    """Surgically edit the document using line numbers."""
    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)
        if not os.path.exists(full_path):
            continue

        print(f"  🤖 AI Surgically Editing: {doc_path}...")
        with open(full_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        numbered_lines = [f("{i+1}: {line}") for i, line in enumerate(original_content.split('\n'))]
        numbered_context = "\n".join(numbered_lines)

        if len(original_content) > 100000:
            continue

        prompt = f"""
        Subject: {commit_subject}
        Release Note: {relnote_text}
        
        CODE CHANGES:
        {commit_diff[:4000]}

        Update {doc_path} using this JSON format:
        {{
            "action": "replace" OR "insert_after",
            "line_number": 123,
            "new_text": "Updated sentences."
        }}

        --- CONTENT ---
        {numbered_context}
        """
        try:
            time.sleep(2)
            response = model.generate_content(prompt, generation_config={"temperature": 0.0})
            raw_json = response.text.strip()
            raw_json = re.sub(r'^```(json)?\s*', '', raw_json, flags=re.IGNORECASE)
            raw_json = re.sub(r'\s*```$', '', raw_json).strip()

            update_data = json.loads(raw_json)
            action = update_data.get("action")
            line_num = int(update_data.get("line_number"))
            new_text = update_data.get("new_text")

            lines = original_content.split('\n')
            idx = line_num - 1

            if action == "replace":
                lines[idx] = new_text
            elif action == "insert_after":
                lines.insert(idx + 1, "")
                lines.insert(idx + 2, new_text)
                lines.insert(idx + 3, "")

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))
            print(f"  ✅ Applied {action} at line {line_num}")

        except Exception as e:
            print(f"  ❌ Surgical edit failed for {doc_path}: {e}")

def run_rulebook():
    res = setup_gemini()
    if not res: return
    flash_model, pro_model = res

    if not os.path.exists('weekly_notes.txt'): return
    with open('weekly_notes.txt', 'r', encoding='utf-8') as f:
        raw_data = f.read()

    commits = raw_data.split('COMMIT_DELIMITER\n')[1:]
    all_doc_paths = get_all_doc_paths()

    for commit_block in commits:
        lines = commit_block.strip().split('\n')
        if len(lines) < 3: continue

        commit_hash = lines[0].strip()
        commit_subject = lines[1].strip()
        body = '\n'.join(lines[2:])

        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match: continue

        note = match.group(1).strip()
        if note.lower() in ['none', 'n/a', 'no']: continue

        print(f"\n✅ Processing: {commit_hash[:7]} - {commit_subject}")
        target_docs = find_best_docs_with_gemini(flash_model, commit_subject, note, all_doc_paths)

        if target_docs and len(target_docs) <= 2:
            try:
                commit_diff = subprocess.check_output(
                    ['git', '-C', 'bazel_src', 'show', '--format=%b', commit_hash],
                    text=True
                ).strip()
            except:
                commit_diff = "Diff unavailable."

            rewrite_docs_with_gemini(pro_model, commit_subject, note, commit_diff, target_docs)

if __name__ == "__main__":
    run_rulebook()
