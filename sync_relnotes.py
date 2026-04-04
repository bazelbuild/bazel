import os
import json
import time
import subprocess
import re
from datetime import datetime
import google.generativeai as genai

# --- CONFIGURATION ---
LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'gemini-2.5-pro')

def setup_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("❌ Error: GEMINI_API_KEY environment variable not set.")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(LLM_MODEL_NAME)

def get_all_doc_paths():
    try:
        out = subprocess.check_output(['git', '-C', 'bazel_src', 'ls-files', 'site/en/**/*.md'], text=True)
        return [p for p in out.split('\n') if p and '/versions/' not in p and '/archive/' not in p]
    except: return []

def run_agent_audit(model, commit_hash, subject, note, diff):
    all_paths = get_all_doc_paths()
    paths_str = "\n".join(all_paths)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    instr_path = os.path.join(script_dir, 'PROMPT_INSTRUCTION.txt')
    instructions = "You are a Bazel documentation agent."
    if os.path.exists(instr_path):
        with open(instr_path, 'r', encoding='utf-8') as f:
            instructions = f.read()

    prompt = f"""
    {instructions}

    You are analyzing this commit:
    Hash: {commit_hash}
    Subject: {subject}
    Note: {note}

    --- CODE DIFF ---
    {diff[:4000]}
    -----------------

    AVAILABLE FILES:
    {paths_str}
    """

    try:
        time.sleep(2)
        response = model.generate_content(prompt, generation_config={"temperature": 0.0})
        
        json_match = re.search(r'\{.*\}', response.text.strip(), re.DOTALL)
        if not json_match:
            return "error", "AI did not return valid JSON.", []
        
        data = json.loads(json_match.group(0))

        if data.get("action") == "skip":
            reason = data.get('reason', 'Internal change')
            print(f"  ⏭️ Skipped: {reason}")
            return "skipped", reason, []

        devsite_path = data.get("file_path")
        if not devsite_path:
            return "error", "No file_path provided by AI.", []

        mintlify_path = devsite_path.replace("site/en/", "docs/").replace(".md", ".mdx")
        target_files = [devsite_path, mintlify_path]
        
        target_text = data.get("target_line_text", "").strip()
        edit_type = data.get("edit_type", "insert_after")
        ai_text = data.get("new_content", "")
        reason = data.get("reason", "Feature documented.")

        for doc_path in target_files:
            full_path = os.path.join('bazel_src', doc_path)
            if not os.path.exists(full_path):
                print(f"  ⚠️ {doc_path} not found. Skipping twin.")
                continue

            with open(full_path, 'r', encoding='utf-8') as f:
                lines = f.read().split('\n')

            # Search for target text and track if we are inside a Markdown Code Block
            line_idx = -1
            in_code_block = False
            target_in_code_block = False
            
            for i, line in enumerate(lines):
                if line.strip().startswith('```'):
                    in_code_block = not in_code_block
                if target_text and target_text in line:
                    line_idx = i
                    target_in_code_block = in_code_block
                    break

            # --- SAFETY SHIELD: Code Block Awareness ---
            if line_idx != -1 and edit_type == "insert_after":
                # If target is inside a block, move line_idx down to the closing backticks
                if target_in_code_block:
                    print(f"  🛡️ Safety Shield: Target line is inside a code block in {doc_path}. Moving out.")
                    for j in range(line_idx, len(lines)):
                        if lines[j].strip().startswith('```'):
                            line_idx = j
                            break
            
            # Formatting: Apply MDX escaping
            final_text = ai_text
            if doc_path.endswith(".mdx"):
                final_text = final_text.replace("{", "\\{").replace("}", "\\}")

            # Formatting: Ensure we have vertical space if near a code block
            if line_idx != -1 and edit_type == "insert_after":
                if line_idx < len(lines) and lines[line_idx].strip().startswith('```'):
                    final_text = "\n" + final_text # Add gap after closing block
                elif line_idx + 1 < len(lines) and lines[line_idx + 1].strip().startswith('```'):
                    final_text = final_text + "\n" # Add gap before starting block

            if line_idx != -1:
                # SAFE SUBSTRING MODIFICATION (Fixes paragraph destruction)
                if edit_type == "delete":
                    lines[line_idx] = lines[line_idx].replace(target_text, "")
                    print(f"  🗑️ Deleted matched text in {doc_path}")
                elif edit_type == "replace":
                    lines[line_idx] = lines[line_idx].replace(target_text, final_text)
                    print(f"  🔄 Replaced matched text in {doc_path}")
                else: # insert_after
                    lines.insert(line_idx + 1, "\n" + final_text + "\n")
                    print(f"  ✅ Inserted update into {doc_path}")
                
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
            else:
                # Safe Fallback
                lines.append("\n" + final_text + "\n")
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(lines))
                print(f"  ⚠️ Target line not found in {doc_path}. Appended to bottom.")

        return "processed", reason, target_files
    except Exception as e:
        print(f"  ❌ Agent Error: {e}")
        return "error", str(e), []

def get_git_date(hash):
    try:
        date_str = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '-s', '--format=%cs', hash], text=True).strip()
        return date_str
    except: return "Unknown Date"

def run_rulebook():
    model = setup_client()
    if not model: return

    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path):
        print("❌ weekly_notes.txt not found.")
        return

    with open(log_path, 'r', encoding='utf-8') as f:
        commits_raw = f.read().split('COMMIT_DELIMITER\n')[1:]

    processed_list = []
    skipped_list = []
    error_list = []
    
    first_date = None
    last_date = None

    for block in commits_raw:
        lines = block.strip().split('\n')
        if len(lines) < 3: continue
        
        hash = lines[0].strip()
        subj = lines[1].strip()
        body = '\n'.join(lines[2:])

        # Track dates for PR description
        c_date = get_git_date(hash)
        if not last_date: last_date = c_date
        first_date = c_date

        print(f"\n🚀 Agent Auditing: {hash[:7]} - {subj[:60]}...")
        
        try:
            # -U20 gives the AI 20 lines of context around the code change!
            diff = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '-U20', '--format=', hash], text=True).strip()
            
            status, reason, files = run_agent_audit(model, hash, subj, "", diff)
            if status == "processed":
                processed_list.append(f"- **{hash[:7]}**: {subj}\n  - *Decision:* Documented in `{', '.join(files)}` because {reason}")
            elif status == "skipped":
                skipped_list.append(f"- **{hash[:7]}**: {subj}\n  - *Decision:* Filtered out because {reason}")
            else:
                error_list.append(f"- **{hash[:7]}**: {subj}\n  - *Error:* {reason}")
                
        except Exception as e:
            print(f"  ⚠️ System Error for {hash[:7]}: {e}")

    # Build the Rich PR Report
    print("\n📝 Generating Rich PR Report...")
    with open("pr_report.md", "w", encoding='utf-8') as f:
        f.write("### 📅 Sync Window\n")
        if first_date and last_date:
            f.write(f"Analyzed merged commits from **{first_date}** to **{last_date}**.\n\n")
        else:
            f.write("Analyzed recent merged commits.\n\n")
            
        f.write("### ✅ Documented Features\n")
        if processed_list:
            f.write("\n".join(processed_list) + "\n\n")
        else:
            f.write("*No public-facing features required documentation updates in this run.*\n\n")
            
        f.write("### ⏭️ Filtered / Internal Commits\n")
        f.write("These commits were identified as internal or test-only by the Agent:\n")
        if skipped_list:
            f.write("\n".join(skipped_list) + "\n\n")
        else:
            f.write("*None.*\n\n")

    print(f"\n✅ Finished. {len(processed_list)} documentation updates applied.")

if __name__ == "__main__":
    run_rulebook()
