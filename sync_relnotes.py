import re
import subprocess
import os
import time
import json
from google import genai
from google.genai import types

def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None
    # Initialize the modern client
    return genai.Client(api_key=api_key)

def rewrite_docs_with_gemini(client, commit_subject, relnote_text, commit_diff, target_docs):
    """Uses Gemini to surgically edit the doc based strictly on the code diff."""
    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)

        if not os.path.exists(full_path):
            continue

        print(f"  🤖 AI Surgically Editing: {doc_path}...")

        with open(full_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # Fixed line wrap for numbered_lines
        numbered_lines = [f"{i+1}: {line}" for i, line in enumerate(original_content.split('\n'))]
        numbered_context = "\n".join(numbered_lines)

        if len(original_content) > 100000:
            print(f"  ⚠️ File {doc_path} is too large. Skipping.")
            continue

        prompt = f"""
        You are an expert technical writer updating Bazel documentation.
        Commit Subject: {commit_subject}
        Release Note: {relnote_text}

        --- ACTUAL CODE CHANGES ---
        {commit_diff[:4000]}
        ---------------------------

        Update the document below (with line numbers).
        RULES:
        1. BASE EDIT ONLY ON CODE CHANGES.
        2. MAX 4 to 5 sentences.
        3. Identify exact line number.
        4. Output ONLY JSON: {{"action": "replace"|"insert_after", "line_number": 123, "new_text": "..."}}

        --- DOCUMENT ---
        {numbered_context}
        """

        try:
            time.sleep(2)
            # Modern SDK call with JSON response enforcement
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                ),
            )

            update_data = json.loads(response.text)
            action = update_data.get("action")
            line_num = update_data.get("line_number")
            new_text = update_data.get("new_text")

            if not action or not line_num or not new_text:
                continue

            lines = original_content.split('\n')
            idx = int(line_num) - 1

            if idx < 0 or idx >= len(lines):
                print(f"  ⚠️ Invalid line number ({line_num}) for {doc_path}")
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
            print(f"  ❌ Gemini AI Error for {doc_path}: {e}")

def run_rulebook():
    client = setup_gemini()
    if not client: return

    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path):
        print(f"❌ {log_path} not found.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()

    commits = raw_data.split('COMMIT_DELIMITER\n')[1:]
    actionable_commits = []

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

        # Extract Keywords
        try:
            changed_files_out = subprocess.check_output(
                ['git', '-C', 'bazel_src', 'show', '--name-only', '--format=', commit_hash],
                text=True
            ).strip()
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
                # Fixed line wrap for the gh search command string
                cmd = f"gh search code '{kw}' --repo bazelbuild/bazel --extension mdx --extension md --limit 5"
                search_out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                
                for line in search_out.strip().split('\n'):
                    if not line: continue
                    path = line.split(':')[0].strip()
                    if '/versions/' in path or '/archive/' in path: continue
                    if ('docs/' in path or 'site/en/' in path) and path.endswith(('.md', '.mdx')):
                        target_docs.add(path)
            except: pass

        # Fallback Search
        if not target_docs:
            clean_subject = commit_subject.replace('`', '').replace("'", "")
            subject_query = ' '.join(clean_subject.split()[:3])
            try:
                cmd_fb = f"gh search code '{subject_query}' --repo bazelbuild/bazel --extension mdx --extension md --limit 3"
                search_out = subprocess.check_output(cmd_fb, shell=True, text=True, stderr=subprocess.DEVNULL)
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
                # Fixed format to prevent commit msg pollution
                diff = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '--format=', commit_hash], text=True).strip()
            except: diff = "Diff unavailable."

            rewrite_docs_with_gemini(client, commit_subject, note, diff, target_docs)
            actionable_commits.append({"hash": commit_hash})
        else:
            print("⚠️ No docs found. Skipping.")

    if actionable_commits:
        print("\n🎉 Safe documentation rewrite complete.")

if __name__ == "__main__":
    run_rulebook()
