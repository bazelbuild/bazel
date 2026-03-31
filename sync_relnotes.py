import re
import subprocess
import os
import time
import json
import google.generativeai as genai

def setup_gemini():
    """Initializes the Gemini API and returns Flash (mapping) and Pro (writing)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None, None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')

def get_all_doc_paths():
    """Gets a list of all current .md files in the Bazel repo."""
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
    """PHASE 1: Identifies the relevant DevSite file and its Mintlify twin."""
    paths_str = "\n".join(all_paths)
    prompt = f"""
    You are an expert Bazel engineer.
    Commit: {commit_subject}
    Note: {relnote}

    TASK: Find the MOST relevant documentation file for this change.
    Return ONLY a JSON string of the path. Example: "site/en/ref.md"
    FILES:
    {paths_str}
    """
    try:
        time.sleep(1)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        match = re.search(r'site/en/[\w./-]+\.md', response.text.strip())
        if not match: return set()

        devsite_path = match.group(0)
        mintlify_path = devsite_path.replace("site/en/", "docs/").replace(".md", ".mdx")

        final_paths = set()
        if os.path.exists(os.path.join('bazel_src', devsite_path)): final_paths.add(devsite_path)
        if os.path.exists(os.path.join('bazel_src', mintlify_path)): final_paths.add(mintlify_path)
        return final_paths
    except Exception: return set()

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
    """PHASE 2: Performs Multi-line Addition, Replacement, or Deletion."""

    # 1. Generate the concise update text
    prompt_text = f"""
    You are a technical writer. Draft a concise documentation update (max 4 lines).
    Commit: {commit_subject}
    Note: {relnote_text}
    Diff: {commit_diff[:3000]}

    Instruction: If the feature is being REMOVED, draft a note about its removal.
    Return ONLY raw markdown text. No explanations.
    """
    try:
        time.sleep(2)
        response_text = model.generate_content(prompt_text, generation_config={"temperature": 0.0})
        new_text = response_text.text.strip()
        new_text = re.sub(r'^```(?:markdown|mdx)?\s*|\s*```$', '', new_text, flags=re.IGNORECASE).strip()
    except Exception as e:
        print(f"  ❌ Error generating text: {e}")
        return

    # 2. Contextual Range Replacement / Deletion
    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)
        if not os.path.exists(full_path): continue

        with open(full_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        content = "".join(original_lines)

        prompt_placement = f"""
        TASK: Update the document based on this code change.
        DIFF: {commit_diff[:1500]}
        NEW TEXT TO ADD: {new_text}

        CRITICAL RULES:
        1. DELETE: If a feature is removed from the code, find the section describing it and use "delete".
        2. REPLACE: If a feature is updated, find the old section and use "replace".
        3. ADD: If it is new, use "insert_after".
        
        To target a MULTI-LINE section, provide the EXACT 'start_matching_line' and 'end_matching_line'.

        JSON STRUCTURE:
        {{
            "action": "replace" | "insert_after" | "delete",
            "start_matching_line": "Exact verbatim first line of the target section",
            "end_matching_line": "Exact verbatim last line of the target section"
        }}

        --- DOCUMENT CONTENT ({doc_path}) ---
        {content[:10000]}
        """

        try:
            time.sleep(1)
            response = model.generate_content(prompt_placement, generation_config={"temperature": 0.0})
            raw_text = response.text.strip()
            
            # Robust JSON extraction
            start_idx_json = raw_text.find('{')
            end_idx_json = raw_text.rfind('}')
            if start_idx_json == -1 or end_idx_json == -1:
                print(f"  ⚠️ No JSON found for {doc_path}. Fallback: Appending.")
                original_lines.append("\n" + new_text + "\n")
            else:
                update = json.loads(raw_text[start_idx_json:end_idx_json+1])
                action = update.get('action', 'insert_after')
                start_match = update.get('start_matching_line', "").strip()
                end_match = update.get('end_matching_line', start_match).strip()

                # Locate line indices for the range
                start_line_idx, end_line_idx = -1, -1
                for i, line in enumerate(original_lines):
                    if start_match and start_match in line: start_line_idx = i
                    if end_match and end_match in line: end_line_idx = i
                    if start_line_idx != -1 and end_line_idx != -1: break

                # Apply MDX escaping
                file_text = new_text
                if doc_path.endswith('.mdx'):
                    file_text = file_text.replace('{', '\\{').replace('}', '\\}')

                if start_line_idx != -1:
                    # If end_line_idx is before start (AI error), reset it to start
                    if end_line_idx < start_line_idx: end_line_idx = start_line_idx

                    if action == "delete":
                        print(f"  🗑️ Deleting lines {start_line_idx+1} to {end_line_idx+1} in {doc_path}")
                        del original_lines[start_line_idx : end_line_idx + 1]
                    elif action == "replace":
                        print(f"  🔄 Replacing range in {doc_path}")
                        original_lines[start_line_idx : end_line_idx + 1] = [file_text + "\n"]
                    else: # insert_after
                        print(f"  ✅ Inserting after match in {doc_path}")
                        original_lines.insert(start_line_idx + 1, "\n" + file_text + "\n")
                else:
                    if action != "delete": 
                        print(f"  ⚠️ Keyword match failed in {doc_path}. Appending.")
                        original_lines.append("\n" + file_text + "\n")

            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(original_lines)

        except Exception as e:
            print(f"  ❌ Error applying to {doc_path}: {e}")

def run_rulebook():
    flash_model, pro_model = setup_gemini()
    if not flash_model: return

    if not os.path.exists('weekly_notes.txt'):
        print("Error: weekly_notes.txt not found.")
        return

    all_doc_paths = get_all_doc_paths()
    with open('weekly_notes.txt', 'r', encoding='utf-8') as f:
        commits_data = f.read().split('COMMIT_DELIMITER\n')[1:]

    for commit_block in commits_data:
        lines = commit_block.strip().split('\n')
        if len(lines) < 3: continue

        commit_hash, commit_subject = lines[0].strip(), lines[1].strip()
        body = '\n'.join(lines[2:])

        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match: continue
        
        # Robust Filtering for "None.", "N/A", etc.
        note = match.group(1).strip()
        clean_note = re.sub(r'[.*`#\s]+$', '', note).lower().strip()
        if clean_note in ['none', 'n/a', 'no', '']: continue

        print(f"\n🚀 Processing: {commit_hash[:7]} - {commit_subject[:50]}...")
        target_docs = find_best_docs_with_gemini(flash_model, commit_subject, note, all_doc_paths)

        if target_docs:
            print(f"  🎯 Target Pair: {list(target_docs)}")
            try:
                diff = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '--format=', commit_hash], text=True).strip()
                rewrite_docs_with_gemini(pro_model, commit_subject, note, diff, target_docs)
            except Exception as e:
                print(f"  ⚠️ Error: {e}")
        else:
            print("  ⏭️ No matching documentation pair found.")

if __name__ == "__main__":
    run_rulebook()
