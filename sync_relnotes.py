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
    # Flash for mapping, Pro for high-quality writing
    return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')

def get_all_doc_paths():
    """Gets a list of all current .md files in the repo. (Python handles .mdx twins)"""
    try:
        out = subprocess.check_output(
            ['git', '-C', 'bazel_src', 'ls-files', 'site/en/**/*.md'],
            text=True
        )
        paths = [p for p in out.split('\n') if p and '/versions/' not in p and '/archive/' not in p]
        return paths
    except Exception as e:
        print(f"Error getting file list: {e}")
        return []

def find_best_docs_with_gemini(model, commit_subject, relnote, all_paths):
    """PHASE 1: Picks exactly one .md and auto-maps its .mdx twin."""
    paths_str = "\n".join(all_paths)
    prompt = f"Subject: {commit_subject}\nNote: {relnote}\nPick EXACTLY one file path from this list: {paths_str}\nReturn ONLY the path."
    try:
        time.sleep(1)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        raw_text = response.text.strip()

        match = re.search(r'site/en/[a-zA-Z0-9_/-]+\.md', raw_text)
        if not match: return set()

        devsite_path = match.group(0)
        mintlify_path = devsite_path.replace("site/en/", "docs/").replace(".md", ".mdx")

        final_paths = set()
        if os.path.exists(os.path.join('bazel_src', devsite_path)): 
            final_paths.add(devsite_path)
        if os.path.exists(os.path.join('bazel_src', mintlify_path)):
            final_paths.add(mintlify_path)

        return final_paths
    except Exception: 
        return set()

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
    """PHASE 2: Writes the update ONCE, then applies it safely using string matching."""

    # 1. Generate the text ONCE to ensure consistency
    prompt_text = f"Draft a 4-line doc update for: {commit_subject}\nNote: {relnote_text}\nDiff: {commit_diff[:2000]}\nRULES: MAX 4 lines. RAW markdown only."

    try:
        time.sleep(2)
        response_text = model.generate_content(prompt_text, generation_config={"temperature": 0.0})
        new_text = response_text.text.strip()
        new_text = re.sub(r'^```(?:markdown|mdx)?\s*|\s*```$', '', new_text, flags=re.IGNORECASE).strip()
        new_text = "\n".join(new_text.split('\n')[:4]) # Strict 4-line limit
    except Exception as e:
        print(f"  ❌ Error generating text: {e}")
        return False

    success = False
    # 2. Apply that exact text to both files via exact string matching
    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)
        if not os.path.exists(full_path): continue

        with open(full_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        if len(original_lines) > 2500:
             print(f"  ⚠️ Skipping {doc_path}: File too large.")
             continue

        content = "".join(original_lines)
        prompt_placement = f"""
        Document: {doc_path}
        TEXT TO ADD: {new_text}

        RULES:
         1. If this new text REPLACES old info, provide the EXACT old sentence from the doc.
         2. If this is NEW info, set action to 'insert_after' and provide the EXACT existing line it follows.
         3. Output ONLY JSON: {{"action": "replace"|"insert_after", "exact_old_line": "..."}}

        DOC CONTENT:
        {content[:8000]}
        """

        try:
            time.sleep(1)
            response = model.generate_content(prompt_placement, generation_config={"temperature": 0.0})
            json_match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
            if not json_match: continue
            
            update = json.loads(json_match.group(0))
            action = update.get("action", "insert_after")
            old_line = update.get("exact_old_line", "").strip()

            # Escape for MDX
            file_text = new_text.replace('{', '\\{') if doc_path.endswith('.mdx') else new_text

            # Find matching line index
            line_idx = -1
            for i, line in enumerate(original_lines):
                if old_line in line:
                    line_idx = i
                    break

            if line_idx == -1:
                print(f"  ⚠️ Match not found in {doc_path}. Fallback to append.")
                original_lines.append("\n" + file_text + "\n")
            else:
                if action == "replace":
                    original_lines[line_idx] = file_text + "\n"
                    print(f"  ✅ Safely replaced line in {doc_path}")
                else:
                    original_lines.insert(line_idx + 1, "\n" + file_text + "\n")
                    print(f"  ✅ Safely inserted in {doc_path}")

            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(original_lines)
            success = True

        except Exception as e:
            print(f"  ❌ Error applying to {doc_path}: {e}")

    return success

def run_rulebook():
    flash_model, pro_model = setup_gemini()
    if not flash_model: return
    
    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path): return

    all_doc_paths = get_all_doc_paths()
    with open(log_path, 'r', encoding='utf-8') as f:
        commits = f.read().split('COMMIT_DELIMITER\n')[1:]

    processed_list = []
    for commit_block in commits:
        lines = commit_block.strip().split('\n')
        if len(lines) < 3: continue
        commit_hash, commit_subject = lines[0].strip(), lines[1].strip()
        body = '\n'.join(lines[2:])

        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match or match.group(1).lower().strip() in ['none', 'n/a']: continue

        print(f"\n🚀 Processing: {commit_hash[:7]} - {commit_subject[:50]}...")
        target_docs = find_best_docs_with_gemini(flash_model, commit_subject, match.group(1).strip(), all_doc_paths)

        if target_docs:
            print(f"  🎯 Target Pair: {list(target_docs)}")
            try:
                diff = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '--format=', commit_hash], text=True).strip()
                if rewrite_docs_with_gemini(pro_model, commit_subject, match.group(1).strip(), diff, target_docs):
                    processed_list.append(f"- {commit_hash[:7]}: {commit_subject}")
            except Exception as e: 
                print(f"  ⚠️ Error: {e}")

    if processed_list:
        with open("processed_commits.txt", "w") as f: 
            f.write("\n".join(processed_list))
        print(f"\n✅ {len(processed_list)} commits processed and saved.")

if __name__ == "__main__":
    run_rulebook()
