import re
import subprocess
import os
import time
import json
import google.generativeai as genai

def setup_gemini():
    """Initializes the Gemini API and returns two models."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None, None
    
    genai.configure(api_key=api_key)
    # Flash is faster for mapping; Pro is smarter for writing/placement
    return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')

def get_all_doc_paths():
    """Gets a list of all current .md files in the Bazel repo."""
    try:
        out = subprocess.check_output(
            ['git', '-C', 'bazel_src', 'ls-files', 'site/en/**/*.md'],
            text=True
        )
        # Filter out versions/archive to avoid updating old documentation
        paths = [p for p in out.split('\n') if p and '/versions/' not in p and '/archive/' not in p]
        return paths
    except Exception as e:
        print(f"Error getting file list: {e}")
        return []

def find_best_docs_with_gemini(model, commit_subject, relnote, all_paths):
    """PHASE 1: Picks exactly one .md file and maps its .mdx twin."""
    paths_str = "\n".join(all_paths)
    prompt = f"""
    You are an expert Bazel engineer.
    Commit: {commit_subject}
    Release Note: {relnote}

    TASK: Find the MOST relevant documentation file for this change.
    Return ONLY a JSON string of the path. Example: "site/en/ref.md"
    If no relevant docs exist, return "".

    --- FILES ---
    {paths_str}
    """
    try:
        time.sleep(1) # Rate limit safety
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        match = re.search(r'site/en/[\w./-]+\.md', response.text.strip())
        if not match:
            return set()

        devsite_path = match.group(0)
        # Map to Mintlify twin
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
    """PHASE 2: Writes the update and applies it safely to files."""

    # 1. Generate the concise update text
    prompt_text = f"""
    You are a technical writer. Draft a MINIMAL note for the Bazel documentation based on this commit.
    Commit: {commit_subject}
    Note: {relnote_text}
    Diff: {commit_diff[:3000]}

    CRITICAL RULES:
    1. Write a MAXIMUM of 4 lines.
    2. If a feature is replaced/removed, describe the NEW state.
    3. Return ONLY the raw markdown text. No explanations.
    """

    try:
        time.sleep(2) # Rate limit safety for Pro model
        response_text = model.generate_content(prompt_text, generation_config={"temperature": 0.0})
        new_text = response_text.text.strip()
        # Clean markdown formatting if AI added it
        new_text = re.sub(r'^```(?:markdown|mdx)?\s*', '', new_text, flags=re.IGNORECASE)
        new_text = re.sub(r'\s*```$', '', new_text).strip()
    except Exception as e:
        print(f"  ❌ Error generating text: {e}")
        return

    # 2. Surgical Placement / Replacement
    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)
        if not os.path.exists(full_path): continue

        with open(full_path, 'r', encoding='utf-8') as f:
            original_lines = f.readlines()

        # Add line numbers for the AI to reference
        numbered_content = "".join([f"{i+1}: {line}" for i, line in enumerate(original_lines)])

        prompt_placement = f"""
        TASK: Update the document to reflect this change.
        DIFF: {commit_diff[:2000]}
        NEW TEXT TO ADD: {new_text}

        CRITICAL RULES:
        1. Use "replace" if the Diff shows a feature is REMOVED or REPLACED.
        2. Use "insert_after" if this is a brand new addition.
        3. Return ONLY JSON:
           {{"action": "replace", "start_line": 10, "end_line": 12}}
           OR
           {{"action": "insert_after", "line_number": 45}}

        --- DOCUMENT ({doc_path}) ---
        {numbered_content[:8000]}
        """

        try:
            time.sleep(1)
            response = model.generate_content(prompt_placement, generation_config={"temperature": 0.0})
            raw_text = response.text.strip()
            
            # FIX: Extract JSON from between the first and last curly braces
            start_idx = raw_text.find('{')
            end_idx = raw_text.rfind('}')
            if start_idx == -1 or end_idx == -1:
                print(f"  ⚠️ No valid JSON found for {doc_path}")
                continue
            
            json_str = raw_text[start_idx:end_idx+1]
            update = json.loads(json_str)

            # Apply MDX escaping for Mintlify files
            file_text = new_text
            if doc_path.endswith('.mdx'):
                file_text = file_text.replace('{', '\\{').replace('}', '\\}')

            # Perform the update
            if update.get('action') == "replace":
                start = int(update.get('start_line', 1)) - 1
                end = int(update.get('end_line', start + 1))
                if 0 <= start < len(original_lines):
                    print(f"  🔄 Replacing lines {start+1} to {end} in {doc_path}")
                    original_lines[start:end] = [file_text + "\n"]
            else: # insert_after
                line_idx = int(update.get('line_number', len(original_lines))) - 1
                if 0 <= line_idx < len(original_lines):
                    print(f"  ✅ Inserting after line {line_idx+1} in {doc_path}")
                    original_lines.insert(line_idx + 1, "\n" + file_text + "\n")
                else:
                    original_lines.append("\n" + file_text + "\n")

            with open(full_path, 'w', encoding='utf-8') as f:
                f.writelines(original_lines)

        except Exception as e:
            print(f"  ❌ Error applying to {doc_path}: {e}")

def run_rulebook():
    flash_model, pro_model = setup_gemini()
    if not flash_model: return

    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    all_doc_paths = get_all_doc_paths()

    with open(log_path, 'r', encoding='utf-8') as f:
        # Split by the custom delimiter used in the git log command
        commits_data = f.read().split('COMMIT_DELIMITER\n')[1:]

    for commit_block in commits_data:
        lines = commit_block.strip().split('\n')
        if len(lines) < 3: continue

        commit_hash, commit_subject = lines[0].strip(), lines[1].strip()
        body = '\n'.join(lines[2:])

        # Find the RELNOTES line
        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match: continue
        
        # FIX: Robust Filtering for "None.", "N/A", "none", etc.
        note = match.group(1).strip()
        # Clean the note: remove trailing periods, asterisks, backticks, and whitespace
        clean_note = re.sub(r'[.*`#\s]+$', '', note).lower().strip()
        
        if clean_note in ['none', 'n/a', 'no', '']: 
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
                # Phase 2: Rewrite and Insert
                rewrite_docs_with_gemini(pro_model, commit_subject, note, diff, target_docs)
            except Exception as e:
                print(f"  ⚠️ Error fetching diff: {e}")
        else:
            print("  ⏭️ No matching documentation pair found.")

if __name__ == "__main__":
    run_rulebook()
