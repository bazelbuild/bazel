import re
import subprocess
import os
import time
import json
import google.generativeai as genai

# --- 1. SETUP ---
def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        return None
    genai.configure(api_key=api_key)
    # Using Pro for complex rule-following and audit logic
    return genai.GenerativeModel('gemini-2.5-pro')

def get_all_doc_paths():
    """Rule 2: Get all DevSite files."""
    try:
        out = subprocess.check_output(['git', '-C', 'bazel_src', 'ls-files', 'site/en/**/*.md'], text=True)
        return [p for p in out.split('\n') if p and '/versions/' not in p and '/archive/' not in p]
    except: return []

# --- 2. THE AGENT AUDIT LOGIC ---
def run_agent_audit(model, commit_hash, subject, note, diff):
    """The AI Agent logic following your 6 strict rules."""
    
    # Get master list for Rule 2 (Mapping)
    all_paths = get_all_doc_paths()
    paths_str = "\n".join(all_paths)

    prompt = f"""
    You are a highly intelligent Bazel Documentation Auditor.
    You MUST follow these 6 rules in order:

    1. FILTER: Is this commit a public-facing feature? (Subject: {subject}, Note: {note}). 
       If internal/test-only, return {{"action": "skip", "reason": "internal"}}.
    2. MAP: Identify the MOST relevant DevSite file from this list:
       {paths_str}
    3. ANALYZE: Study the Git Diff below and find exactly where documentation should be added or changed.
    4. REMOVALS: If code was removed in the diff, identify the corresponding lines in the doc to remove or update.
    5. ADDITION LIMIT: Write a MAXIMUM of 3-4 technical lines for the update.
    6. DELETION LIMIT: Remove a MAXIMUM of 2 lines only if justified by code removal.

    GIT DIFF:
    {diff[:4000]}

    OUTPUT INSTRUCTIONS:
    Return ONLY a JSON object with these exact keys:
    - "action": "update" or "skip"
    - "file_path": "site/en/..." (The DevSite path)
    - "edit_type": "insert_after" or "replace" or "delete"
    - "target_line_text": "Exact verbatim line from the document to target"
    - "new_content": "Your concise 3-4 lines of documentation"
    """

    try:
        time.sleep(2)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        
        # Robust JSON extraction
        json_match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
        if not json_match:
            print(f"  ⚠️ AI did not return valid JSON. Skipping.")
            return False
        
        data = json.loads(json_match.group(0))

        if data.get("action") == "skip":
            print(f"  ⏭️ Skipped: {data.get('reason', 'internal')}")
            return False

        devsite_path = data["file_path"]
        # Calculate Mintlify twin: site/en/run/bazelrc.md -> docs/run/bazelrc.mdx
        mintlify_path = devsite_path.replace("site/en/", "docs/").replace(".md", ".mdx")
        
        target_files = [devsite_path, mintlify_path]
        success = False

        for doc_path in target_files:
            full_path = os.path.join('bazel_src', doc_path)
            if not os.path.exists(full_path):
                print(f"  ⚠️ {doc_path} not found. Skipping twin.")
                continue

            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')

            # Find the line the AI targeted
            target_text = data["target_line_text"].strip()
            line_idx = -1
            for i, line in enumerate(lines):
                if target_text in line:
                    line_idx = i
                    break

            # Handle MDX escaping for Mintlify
            final_content = data["new_content"]
            if doc_path.endswith(".mdx"):
                final_content = final_content.replace("{", "\\{").replace("}", "\\}")

            if line_idx != -1:
                # Rule 5 & 6 Enforcement
                if data["edit_type"] == "delete":
                    del lines[line_idx]
                    print(f"  🗑️ Deleted line in {doc_path}")
                elif data["edit_type"] == "replace":
                    lines[line_idx] = final_content
                    print(f"  🔄 Replaced line in {doc_path}")
                else: # insert_after
                    lines.insert(line_idx + 1, "\n" + final_content + "\n")
                    print(f"  ✅ Inserted update into {doc_path}")
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                success = True
            else:
                # Safe Fallback: If no match, append to bottom to preserve work
                lines.append("\n" + final_content + "\n")
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print(f"  ⚠️ Target line not found in {doc_path}. Appended to bottom.")
                success = True

        return success
    except Exception as e:
        print(f"  ❌ Agent Execution Error: {e}")
        return False

# --- 3. MAIN RUNNER ---
def run_rulebook():
    model = setup_gemini()
    if not model: return

    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path):
        print("❌ weekly_notes.txt not found.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        commits_raw = f.read().split('COMMIT_DELIMITER\n').slice(1)

    processed_list = []
    for block in commits_raw:
        lines = block.strip().split('\n')
        if len(lines) < 3: continue
        
        hash = lines[0].strip()
        subj = lines[1].strip()
        body = '\n'.join(lines[2:])

        # Extract RELNOTES
        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match: continue
        note = match.group(1).strip()
        if note.lower() in ['none', 'n/a', 'no', '']: continue

        print(f"\n🚀 Agent Auditing: {hash[:7]} - {subj[:60]}...")
        
        try:
            # Fetch Diff
            diff = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '--format=', hash], text=True).strip()
            
            # Execute the 6-rule Agent Audit
            if run_agent_audit(model, hash, subj, note, diff):
                processed_list.append(f"- {hash[:7]}: {subj}")
        except Exception as e:
            print(f"  ⚠️ System Error for {hash[:7]}: {e}")

    # Save findings for the PR description
    if processed_list:
        with open("processed_commits.txt", "w") as f:
            f.write("\n".join(processed_list))
        print(f"\n✅ Finished. {len(processed_list)} documentation updates applied.")

if __name__ == "__main__":
    run_rulebook()
