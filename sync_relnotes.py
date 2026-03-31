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
    # Flash for fast path selection, Pro for massive file rewrites
    return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')

def get_all_doc_paths():
    """Gets a list of all current .md and .mdx files in the repo."""
    try:
        # Combined patterns for git ls-files
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
    You are an expert Bazel engineer. A new commit was merged:
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
        time.sleep(1)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        raw_json = response.text.strip()
        
        # Clean markdown code blocks from the AI response
        raw_json = re.sub(r'^```(?:json)?\s*', '', raw_json, flags=re.IGNORECASE)
        raw_json = re.sub(r'\s*```$', '', raw_json).strip()
        
        selected_paths = json.loads(raw_json)
        return set(selected_paths)
    except Exception as e:
        print(f"  ⚠️ AI File Selection Error: {e}")
        return set()

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
    """PHASE 2: Re-writes the ENTIRE file with 3-4 new lines added."""

    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)

        if not os.path.exists(full_path):
            continue

        print(f"  🤖 AI Updating Entire File: {doc_path}...")

        with open(full_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        if len(original_content) > 60000:
            print(f"  ⚠️ File {doc_path} is too massive. Skipping.")
            continue

        prompt = f"""
        You are an expert technical writer updating Bazel documentation.

        Commit Subject: {commit_subject}
        Release Note: {relnote_text}
        Actual Code Changes:
        {commit_diff[:3000]}

        Please update the following documentation file to include this new information.

        CRITICAL RULES:
         1. Keep the exact same formatting and headers.
         2. BE BRIEF. Add a MAXIMUM of 3 to 4 sentences based on the code changes.
         3. Integrate the note naturally into the most relevant section.
         4. If this is an MDX file, escape bare braces ({{ -> \\{{).
         5. Return the ENTIRE updated file content exactly as it should be saved.
         6. DO NOT use omission placeholders like "...". You MUST output the full file.
         7. Do not wrap in markdown code blocks.

        --- EXISTING DOCUMENTATION CONTENT ({doc_path}) ---
        {original_content}
        """

        try:
            time.sleep(2) 
            response = model.generate_content(prompt, generation_config={
                "temperature": 0.0,
                "max_output_tokens": 8192
            })
            new_content = response.text.strip()

            # Remove AI-generated code blocks if present
            new_content = re.sub(r'^```(?:markdown|mdx)?\s*', '', new_content, flags=re.IGNORECASE)
            new_content = re.sub(r'\s*```$', '', new_content).strip()

            # Safety Check: Did the AI delete significant parts of the file?
            if len(new_content) < len(original_content) * 0.7:
                print(f"  ❌ Safety Abort: AI truncated {doc_path}. Changes discarded.")
                continue

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)

            print(f"  ✅ Successfully updated and saved {doc_path}")

        except Exception as e:
            print(f"  ❌ Gemini Error for {doc_path}: {e}")

def run_rulebook():
    models = setup_gemini()
    if not models: return
    flash_model, pro_model = models

    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path):
        print(f"Error: {log_path} not found.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()

    commits = raw_data.split('COMMIT_DELIMITER\n')[1:]
    actionable_commits = []

    all_doc_paths = get_all_doc_paths()
    print(f"📚 Loaded {len(all_doc_paths)} documentation files.")

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

        if clean_note in ['none', 'none.', 'n/a', 'na', 'no'] or "<reason>" in note:
            continue

        print(f"\n✅ Processing: {commit_hash[:7]} - {commit_subject}")

        # PHASE 1: Find best docs
        target_docs = find_best_docs_with_gemini(flash_model, commit_subject, note, all_doc_paths)

        if target_docs:
            print(f"  🎯 AI selected: {target_docs}")
            try:
                commit_diff = subprocess.check_output(
                    ['git', '-C', 'bazel_src', 'show', '--format=', commit_hash], 
                    text=True
                ).strip()
            except:
                commit_diff = "Diff unavailable."

            # PHASE 2: Rewrite entire file
            rewrite_docs_with_gemini(pro_model, commit_subject, note, commit_diff, target_docs)
            actionable_commits.append({"hash": commit_hash})
        else:
            print("  ⚠️ AI could not confidently select a pair. Skipping.")

    if actionable_commits:
        print("\n🎉 Documentation rewrite complete.")
    else:
        print("\n😴 No actionable documentation updates found.")

if __name__ == "__main__":
    run_rulebook()
