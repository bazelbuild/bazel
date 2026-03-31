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
    # Using 1.5 models as they are currently standard
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
    prompt = f"""
    You are an expert Bazel engineer.
    Commit: {commit_subject}
    Release Note: {relnote}

    TASK:
    Find the MOST relevant DevSite documentation file for this change.
    You MUST return EXACTLY ONE file starting with 'site/en/' and ending in '.md'.

    Return ONLY a JSON string of the path. Example: "site/en/ref.md"
    If no relevant docs exist, return "".

    --- FILES ---
    {paths_str}
    """
    try:
        time.sleep(1)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        raw_text = response.text.strip()

        match = re.search(r'site/en/[\w./-]+\.md', raw_text)
        if not match:
            return set()

        devsite_path = match.group(0)
        mintlify_path = devsite_path.replace("site/en/", "docs/").replace(".md", ".mdx")

        final_paths = set()
        if os.path.exists(os.path.join('bazel_src', devsite_path)):
            final_paths.add(devsite_path)
        if os.path.exists(os.path.join('bazel_src', mintlify_path)):
            final_paths.add(mintlify_path)

        return final_paths
    except:
        return set()

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
    """PHASE 2: Writes the update ONCE, then applies it to both files."""

    # 1. Generate the text ONCE to ensure consistency between .md and .mdx
    prompt_text = f"""
    You are a technical writer. Draft a MINIMAL note for the Bazel documentation based on this commit.

    Commit: {commit_subject}
    Note: {relnote_text}
    Diff: {commit_diff[:2000]}

    CRITICAL RULES:
    1. Write a MAXIMUM of 4 lines. DO NOT write paragraphs.
    2. Be extremely concise. Use one bullet point or two short sentences.
    3. Return ONLY the raw markdown text. No explanations.
    """

    try:
        time.sleep(2)
        response_text = model.generate_content(prompt_text, generation_config={"temperature": 0.0})
        new_text = response_text.text.strip()
        # Clean markdown formatting if AI added it
        new_text = re.sub(r'^```(?:markdown|mdx)?\s*', '', new_text, flags=re.IGNORECASE)
        new_text = re.sub(r'\s*```$', '', new_text).strip()

        # Hard enforcement of 4-line limit in Python
        lines_to_add = new_text.split('\n')
        if len(lines_to_add) > 4:
            print(f"  ⚠️ AI tried to add {len(lines_to_add)} lines. Truncating to 4.")
            new_text = "\n".join(lines_to_add[:4])
            lines_to_add = lines_to_add[:4]

    except Exception as e:
        print(f"  ❌ Error generating text: {e}")
        return

    # 2. Ask where to put that text in each file individually
    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)
        if not os.path.exists(full_path): continue

        with open(full_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        if len(original_lines) > 3000:
             print(f"  ⚠️ Skipping {doc_path}: File too large.")
             continue

        numbered_content = "".join([f"{i+1}: {line}" for i, line in enumerate(original_lines)])

        prompt_placement = f"""
        Where should this text be inserted in the document below?

        TEXT TO INSERT:
        {new_text}

        CRITICAL RULES:
         1. ALWAYS prefer "insert_after". ONLY use "replace" if the old line is outdated.
         2. Return ONLY JSON:
        {{
            "action": "insert_after" or "replace",
            "line_number": 123
        }}

        --- DOCUMENT ({doc_path}) ---
        {numbered_content}
        """

        try:
            time.sleep(1)
            response = model.generate_content(prompt_placement, generation_config={"temperature": 0.0})
            raw_json = response.text.strip()
            raw_json = re.sub(r'^```(?:json)?\s*', '', raw_json, flags=re.IGNORECASE)
            raw_json = re.sub(r'\s*```$', '', raw_json).strip()

            try:
                update = json.loads(raw_json)
            except json.JSONDecodeError:
                match = re.search(r'\{.*\}', raw_json, re.DOTALL)
                if match: 
                    update = json.loads(match.group(0))
                else: 
                    raise ValueError("Invalid JSON")

            line_idx = int(update['line_number']) - 1

            # Apply MDX escaping only if it's the .mdx file
            file_text = new_text
            if doc_path.endswith('.mdx'):
                file_text = file_text.replace('{', '\\{').replace('}', '\\}')

            if update['action'] == "replace":
                if 0 <= line_idx < len(original_lines):
                    original_lines[line_idx] = file_text + "\n"
            else: # insert_after
                if 0 <= line_idx < len(original_lines):
                    original_lines.insert(line_idx + 1, "\n" + file_text + "\n")

            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(original_lines)

            print(f"  ✅ Surgical Update: {doc_path} ({len(lines_to_add)} lines synced)")

        except Exception as e:
            print(f"  ❌ Error applying to {doc_path}: {e}")

def run_rulebook():
    models = setup_gemini()
    if not models: return
    flash_model, pro_model = models

    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    all_doc_paths = get_all_doc_paths()

    with open(log_path, 'r', encoding='utf-8') as f:
        commits_data = f.read().split('COMMIT_DELIMITER\n')[1:]

    for commit_block in commits_data:
        lines = commit_block.strip().split('\n')
        if len(lines) < 3: continue

        commit_hash, commit_subject = lines[0].strip(), lines[1].strip()
        body = '\n'.join(lines[2:])

        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match: continue
        note = match.group(1).strip()
        
        # Filter out "None" or "N/A" notes
        if re.sub(r'[*`]', '', note).lower().strip() in ['none', 'n/a', 'no']: 
            continue

        print(f"\n🚀 Processing: {commit_hash[:7]} - {commit_subject[:50]}...")

        # Phase 1: Identify the file pair
        target_docs = find_best_docs_with_gemini(flash_model, commit_subject, note, all_doc_paths)

        if target_docs:
            print(f"  🎯 Target Pair: {list(target_docs)}")
            try:
                diff = subprocess.check_output(
                    ['git', '-C', 'bazel_src', 'show', '--format=', commit_hash], 
                    text=True
                ).strip()
            except:
                diff = "No diff available."

            # Phase 2: Rewrite
            rewrite_docs_with_gemini(pro_model, commit_subject, note, diff, target_docs)
        else:
            print("  ⏭️ No matching documentation pair found.")

if __name__ == "__main__":
    run_rulebook()
