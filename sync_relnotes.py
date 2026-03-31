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
    # Return both models (Flash for search, Pro for careful writing)
    return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
    """Uses Gemini to surgically edit the doc based strictly on the code diff."""
    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)

        if not os.path.exists(full_path):
            continue

        print(f"  🤖 AI Surgically Editing: {doc_path}...")

        with open(full_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Added line numbers so Gemini can target exactly where to make the change
        numbered_lines = [f"{i+1}: {line}" for i, line in enumerate(original_content.split('\n'))]
        numbered_context = "\n".join(numbered_lines)

        if len(original_content) > 100000:
            print(f"  ⚠️ File {doc_path} is too large. Skipping.")
            continue

        prompt = f"""
        You are an expert technical writer updating Bazel documentation.
        Commit Subject: {commit_subject}
        Release Note: {relnote_text}

        ACTUAL CODE CHANGES:
        {commit_diff[:4000]}

        Update the document below (provided with line numbers) to reflect these changes.
        RULES:
        1. BASE YOUR EDIT ONLY ON THE CODE CHANGES.
        2. Write a MAXIMUM of 4 to 5 clear sentences.
        3. Identify the exact line number where this update should be applied.
        4. Output your response in EXACTLY this JSON format:
        {{
            "action": "replace" OR "insert_after",
            "line_number": 123,
            "new_text": "The updated or new sentences."
        }}

        --- NUMBERED DOCUMENT CONTENT ---
        {numbered_context}
        """

        try:
            time.sleep(2)
            response = model.generate_content(prompt, generation_config={"temperature": 0.0, "max_output_tokens": 1024})
            text = response.text.strip()

            # Robust JSON extraction (finds {} even if surrounded by text)
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if not json_match:
                print(f"  ⚠️ AI did not return JSON for {doc_path}")
                continue

            update_data = json.loads(json_match.group(0))
            action = update_data.get("action")
            line_num = int(update_data.get("line_number"))
            new_text = update_data.get("new_text")

            # APPLY THE EDIT SURGICALLY
            lines = original_content.split('\n')
            idx = line_num - 1

            if idx < 0 or idx >= len(lines):
                print(f"  ⚠️ AI suggested an invalid line number ({line_num})")
                continue

            if action == "replace":
                lines[idx] = new_text
                print(f"  ✅ Replaced line {line_num} in {doc_path}")
            elif action == "insert_after":
                lines.insert(idx + 1, "")
                lines.insert(idx + 2, new_text)
                lines.insert(idx + 3, "")
                print(f"  ✅ Inserted after line {line_num} in {doc_path}")

            with open(full_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(lines))

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

    commits = raw_data.split('COMMIT_DELIMITER\n')[1:]
    actionable_commits = []

    for commit_block in commits:
        lines = commit_block.strip().split('\n')
        if len(lines) < 3: continue
        commit_hash, commit_subject = lines[0].strip(), lines[1].strip()
        body = '\n'.join(lines[2:])

        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match: continue
        note = match.group(1).strip()
        if note.lower() in ['none', 'n/a', 'na'] or "<reason>" in note: continue

        print(f"\n✅ Processing: {commit_hash[:7]} - {commit_subject}")

        # Extract Keywords
        try:
            cmd_files = ['git', '-C', 'bazel_src', 'show', '--name-only', '--format=', commit_hash]
            changed_files_out = subprocess.check_output(cmd_files, text=True).strip()
            changed_files = [f for f in changed_files_out.split('\n') if f]
        except: continue

        keywords = set()
        for f in changed_files:
            if f.endswith(('.java', '.cc', '.bzl')):
                filename = f.split('/')[-1].split('.')[0]
                if len(filename) > 3: keywords.add(filename)

        target_docs = set()
        # Primary Search
        for kw in list(keywords)[:5]:
            try:
                # Fixed string formatting and shell command
                search_cmd = f"gh search code '{kw}' --repo bazelbuild/bazel --extension mdx --extension md --limit 5"
                search_out = subprocess.check_output(search_cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                for line in search_out.strip().split('\n'):
                    if not line: continue
                    path = line.split(':')[0].strip()
                    if '/versions/' in path or '/archive/' in path: continue
                    if ('docs/' in path or 'site/en/' in path or 'site/content/en/' in path) and path.endswith(('.md', '.mdx')):
                        target_docs.add(path)
            except: pass

        # Fallback Search
        if not target_docs:
            clean_subject = commit_subject.replace('`', '').replace("'", "")
            subject_query = ' '.join(clean_subject.split()[:3])
            try:
                fb_cmd = f"gh search code '{subject_query}' --repo bazelbuild/bazel --extension mdx --extension md --limit 3"
                search_out = subprocess.check_output(fb_cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                for line in search_out.strip().split('\n'):
                    if not line: continue
                    path = line.split(':')[0].strip()
                    if '/versions/' in path or '/archive/' in path: continue
                    if ('docs/' in path or 'site/' in path) and path.endswith(('.md', '.mdx')):
                        target_docs.add(path)
            except: pass

        if target_docs:
            print(f"📄 Found matching docs: {target_docs}")
            try:
                diff_cmd = ['git', '-C', 'bazel_src', 'show', '--format=', commit_hash]
                commit_diff = subprocess.check_output(diff_cmd, text=True).strip()
            except: commit_diff = "No diff."

            rewrite_docs_with_gemini(pro_model, commit_subject, note, commit_diff, target_docs)
            actionable_commits.append({"hash": commit_hash})

    print("\n🎉 Documentation rewrite run complete.")

if __name__ == "__main__":
    run_rulebook()
