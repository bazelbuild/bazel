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
    # Using 1.5-flash for discovery and 1.5-pro for writing
    return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')

def get_all_doc_paths():
    """Gets a list of all current .md and .mdx files in the repo."""
    try:
        # Get all files tracked by git in bazel_src
        out = subprocess.check_output(['git', '-C', 'bazel_src', 'ls-files', 'site/en/**/*.md', 'docs/**/*.mdx'], text=True)
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

    Below is a list of all documentation files in the Bazel repository.
    Your job is to find the MOST appropriate file to document this change.

    CRITICAL RULES:
    1. You must select EXACTLY ONE DevSite file (starts with 'site/en/' and ends with '.md').
    2. You must select EXACTLY ONE Mintlify file (starts with 'docs/' and ends with '.mdx').
    3. The two files should cover the same topic.
    4. Return ONLY a JSON array containing the two exact file paths. Do not include markdown formatting.
    5. If no file is a good fit, return an empty array [].

    --- AVAILABLE FILES ---
    {paths_str}
    """

    try:
        time.sleep(1) # Prevent rate limits
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        # Clean potential markdown code blocks from response
        raw_json = response.text.strip().replace('```json', '').replace('```', '').strip()
        selected_paths = json.loads(raw_json)
        return set(selected_paths)
    except Exception as e:
        print(f"  ⚠️ Gemini could not determine the best file: {e}")
        return set()

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, target_docs):
    """Uses Gemini Pro to carefully insert 3-4 lines into the chosen files."""
    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)

        if not os.path.exists(full_path):
            continue

        print(f"  🤖 AI Updating: {doc_path}...")

        with open(full_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        if len(original_content) > 100000:
            print(f"  ⚠️ File {doc_path} is too large. Skipping.")
            continue

        prompt = f"""
        You are an expert technical writer updating Bazel documentation.

        Commit Subject: {commit_subject}
        Release Note: {relnote_text}

        Please update the following documentation file to include this new information.

        CRITICAL RULES:
        1. Keep the exact same formatting and headers.
        2. BE BRIEF. Add a MAXIMUM of 3 to 4 sentences.
        3. Integrate the note naturally into the most relevant section.
        4. If this is an MDX file, escape bare braces ({{{{ -> \\{{{{).
        5. Return the ENTIRE updated file content. Do not wrap in markdown code blocks.

        --- EXISTING DOCUMENTATION CONTENT ({doc_path}) ---
        {original_content}
        """

        try:
            time.sleep(3) # Ensure the Pro model has time to process
            response = model.generate_content(prompt, generation_config={"temperature": 0.1, "max_output_tokens": 8192})
            new_content = response.text.strip()
            
            # Remove AI-generated code blocks if present
            new_content = re.sub(r'^```(markdown|mdx)?\s*', '', new_content, flags=re.IGNORECASE)
            new_content = re.sub(r'```\s*$', '', new_content).strip()

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"  ✅ Gemini successfully updated {doc_path}")
        except Exception as e:
            print(f"  ❌ Gemini AI Error for {doc_path}: {e}")

def run_rulebook():
    models = setup_gemini()
    if not models: return
    flash_model, pro_model = models

    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path): return

    with open(log_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()

    commits = raw_data.split('COMMIT_DELIMITER\n')[1:]
    actionable_commits = []

    all_doc_paths = get_all_doc_paths()
    print(f"📚 Loaded {len(all_doc_paths)} documentation files for AI analysis.")

    for commit_block in commits:
        lines = commit_block.strip().split('\n')
        if len(lines) < 3: continue

        commit_hash = lines[0].strip()
        commit_subject = lines[1].strip()
        body = '\n'.join(lines[2:])

        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match: continue

        note = match.group(1).strip()
        clean_note = re.sub(r'[*`]', '', note).lower().strip()

        if clean_note in ['none', 'none.', 'n/a', 'na', 'no'] or "<reason>' here" in note:
            continue

        print(f"\n✅ Processing: {commit_hash[:7]} - {commit_subject}")

        # PHASE 1: Find target files
        target_docs = find_best_docs_with_gemini(flash_model, commit_subject, note, all_doc_paths)

        if target_docs and len(target_docs) >= 1:
            print(f"  🎯 AI selected exact files: {target_docs}")
            # PHASE 2: Rewrite
            rewrite_docs_with_gemini(pro_model, commit_subject, note, target_docs)
            actionable_commits.append({"hash": commit_hash})
        else:
            print("  ⚠️ AI could not confidently select a pair. Skipping.")

    if actionable_commits:
        print("\n🎉 Documentation rewrite complete.")
    else:
        print("\n😴 No actionable documentation updates found.")

if __name__ == "__main__":
    run_rulebook()
