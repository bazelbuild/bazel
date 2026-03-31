import re
import subprocess
import os
import time
import json
import google.generativeai as genai

def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        return None
    genai.configure(api_key=api_key)
    # Flash for fast mapping/scoring, Pro for careful writing
    return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')

def rewrite_docs_with_gemini(flash_model, pro_model, commit_subject, relnote_text, commit_diff, target_docs):
    """Uses Gemini to surgically edit the doc based strictly on the code diff."""

    for doc_path in target_docs:
        full_path = os.path.join('bazel_src', doc_path)

        if not os.path.exists(full_path):
            continue

        print(f"  🤖 AI Surgically Editing: {doc_path}...")

        with open(full_path, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        # Chunking for large files
        chunk_size = 500
        best_chunk_start = 0

        if len(lines) > chunk_size:
            print(f"  🔍 File is large ({len(lines)} lines). Finding the most relevant section...")
            best_score = -1

            for i in range(0, len(lines), chunk_size):
                chunk = "\n".join(lines[i : i + chunk_size])
                eval_prompt = f"""
                Rate how relevant this section of the documentation is to the following commit.
                Commit Subject: {commit_subject}
                Release Note: {relnote_text}

                Return ONLY a number from 0 to 10 (10 being highly relevant).

                --- DOC SECTION ---
                {chunk[:2000]}
                """
                try:
                    time.sleep(0.5)
                    response = flash_model.generate_content(eval_prompt)
                    score_match = re.search(r'\d+', response.text.strip())
                    score = int(score_match.group()) if score_match else 0
                    if score > best_score:
                        best_score = score
                        best_chunk_start = i
                except:
                    continue

        # Extract the chosen section (plus padding)
        start_idx = max(0, best_chunk_start - 50)
        end_idx = min(len(lines), best_chunk_start + chunk_size + 50)
        
        numbered_chunk = [f"{i+1}: {line}" for i, line in enumerate(lines[start_idx:end_idx], start=start_idx)]
        numbered_context = "\n".join(numbered_chunk)

        prompt = f"""
        You are an expert technical writer updating Bazel documentation.
        Commit Subject: {commit_subject}
        Release Note: {relnote_text}

        --- ACTUAL CODE CHANGES ---
        {commit_diff[:4000]}
        ---------------------------

        Update the relevant section below to reflect these code changes.

        CRITICAL RULES:
        1. BASE EDIT ONLY ON CODE CHANGES. 
        2. MAX 4 to 5 sentences.
        3. ALWAYS prefer "insert_after".
        4. ONLY use "replace" if an existing line is factually wrong.
        5. Output EXACTLY this JSON:
        {{
            "action": "replace" OR "insert_after",
            "line_number": 123,
            "new_text": "text"
        }}

        --- RELEVANT DOCUMENT SECTION (Numbered) ---
        {numbered_context}
        """

        try:
            time.sleep(2)
            response = pro_model.generate_content(prompt, generation_config={"temperature": 0.0, "max_output_tokens": 1024})
            
            raw_json = response.text.strip()
            raw_json = re.sub(r'^```(?:json)?\s*', '', raw_json, flags=re.IGNORECASE)
            raw_json = re.sub(r'\s*```$', '', raw_json).strip()

            update_data = json.loads(raw_json)
            action = update_data.get("action")
            line_num = int(update_data.get("line_number"))
            new_text = update_data.get("new_text")

            idx = line_num - 1
            if 0 <= idx < len(lines):
                if action == "replace":
                    lines[idx] = new_text
                    print(f"  ✅ Replacement at line {line_num} in {doc_path}")
                elif action == "insert_after":
                    # Surgical insertion with padding
                    lines.insert(idx + 1, "")
                    lines.insert(idx + 2, new_text)
                    lines.insert(idx + 3, "")
                    print(f"  ✅ Insertion after line {line_num} in {doc_path}")
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
            else:
                print(f"  ⚠️ Invalid line number: {line_num}")

        except Exception as e:
            print(f"  ❌ Gemini/Parse Error for {doc_path}: {e}")

def run_rulebook():
    models = setup_gemini()
    if not models: return
    flash_model, pro_model = models

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

        if clean_note in ['none', 'none.', 'n/a', 'na', 'no'] or "<reason>" in note:
            continue

        print(f"\n✅ Processing: {commit_hash[:7]} - {commit_subject}")

        # Extract Keywords
        try:
            changed_files_out = subprocess.check_output(
                ['git', '-C', 'bazel_src', 'show', '--name-only', '--format=', commit_hash],
                text=True
            ).strip()
            changed_files = [f for f in changed_files_out.split('\n') if f]
        except:
            changed_files = []

        keywords = set()
        for f in changed_files:
            if f.endswith(('.java', '.cc', '.bzl')):
                name = f.split('/')[-1].split('.')[0]
                if len(name) > 3: keywords.add(name)

        target_docs = set()

        # Search using GH CLI
        for kw in list(keywords)[:3]:
            try:
                cmd = f"gh search code '{kw}' --repo bazelbuild/bazel --extension mdx --extension md --limit 5"
                search_out = subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.DEVNULL)
                for line in search_out.strip().split('\n'):
                    if not line: continue
                    path = line.split('\t')[-1] if '\t' in line else line.split(':')[-1]
                    path = path.strip()
                    if any(x in path for x in ['/versions/', '/archive/']): continue
                    if ('docs/' in path or 'site/' in path) and path.endswith(('.md', '.mdx')):
                        target_docs.add(path)
            except:
                continue

        if target_docs:
            print(f"  📄 Found potential docs: {target_docs}")
            try:
                commit_diff = subprocess.check_output(
                    ['git', '-C', 'bazel_src', 'show', '--format=', commit_hash], 
                    text=True
                ).strip()
                rewrite_docs_with_gemini(flash_model, pro_model, commit_subject, note, commit_diff, target_docs)
                actionable_commits.append({"hash": commit_hash})
            except Exception as e:
                print(f"  ❌ Error processing {commit_hash}: {e}")
        else:
            print("  ⚠️ No doc matches found.")

    if actionable_commits:
        print(f"\n🎉 Done. Updated {len(actionable_commits)} commits.")

if __name__ == "__main__":
    run_rulebook()
