import re
import subprocess
import os
import google.generativeai as genai

def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY environment variable not set.")
        return None
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-2.5-flash')

def rewrite_docs_with_gemini(model, commit_subject, relnote_text, target_docs):
    """Uses Gemini API to actually rewrite the documentation files in place."""
    
    for doc_path in target_docs:
        # Strip potential search artifacts from the path (like 'bazelbuild/bazel:')
        clean_path = doc_path.split(':')[-1].strip()
        
        # Construct the absolute path based on the local bazel_src clone
        full_path = os.path.join('bazel_src', clean_path)
        
        if not os.path.exists(full_path):
            print(f"⚠️ Warning: File {full_path} not found locally. Skipping AI rewrite.")
            continue
            
        print(f"🤖 Asking Gemini to update: {clean_path}")
        
        with open(full_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        prompt = f"""
        You are an expert technical writer for the Bazel build system documentation.
        
        A new PR/Commit was just merged:
        Subject: {commit_subject}
        Release Note (Developer Intent): {relnote_text}
        
        I need you to update the existing documentation file to reflect this change.
        Do NOT just append the release note to the bottom. Integrate the information naturally and logically into the most appropriate section of the file.
        
        CRITICAL RULES:
        1. Keep the exact same formatting, tone, and existing headers.
        2. If this is an MDX file, escape bare braces (`{{` -> `\\{{` and `}}` -> `\\}}`).
        3. Convert raw links `<https://...>` to standard `[text](url)` markdown.
        4. Replace HTML comments with JSX comments `{{/* */}}`.
        5. Return ONLY the raw, updated markdown/mdx content. Do not wrap your response in markdown code blocks (e.g., no ```md ... ```).
        
        --- EXISTING DOCUMENTATION CONTENT ---
        {original_content}
        """
        
        try:
            response = model.generate_content(prompt, generation_config={"temperature": 0.1})
            new_content = response.text.strip()
            
            # Strip markdown code blocks if Gemini accidentally added them
            new_content = re.sub(r'^```(markdown|mdx)?\s*', '', new_content, flags=re.IGNORECASE)
            new_content = re.sub(r'```\s*$', '', new_content).strip()
            
            # Write the updated content back to the file
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
                
            print(f"✅ Gemini successfully updated {clean_path}")
        except Exception as e:
            print(f"❌ Gemini AI Error for {clean_path}: {e}")


def run_rulebook():
    # Setup AI Model
    model = setup_gemini()
    if not model:
        return

    # Ensure we are looking for the file in the current working directory
    log_path = 'weekly_notes.txt'
    if not os.path.exists(log_path):
        print(f"❌ {log_path} not found. Skipping discovery.")
        return

    # Use utf-8 to handle special characters in commit logs
    with open(log_path, 'r', encoding='utf-8') as f:
        raw_data = f.read()

    # Split and remove the first empty element
    commits = raw_data.split('COMMIT_DELIMITER')[1:]
    actionable_commits = []

    for commit_block in commits:
        lines = commit_block.strip().split('')
        if len(lines) < 3:
            continue

        commit_hash = lines[0].strip()
        commit_subject = lines[1].strip()
        body = ''.join(lines[2:])

        # RULE 1: Filter out noise and extract intent
        # Handles RELNOTES, RELNOTES:, and RELNOTES[INC]:
        match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
        if not match:
            continue

        note = match.group(1).strip()
        clean_note = re.sub(r'[*`]', '', note).lower().strip()

        # Skip boilerplate or "None" notes
        if clean_note in ['none', 'none.', 'n/a', 'na', 'no'] or "<reason>' here" in note:
            continue

        print(f"
✅ Processing: {commit_hash[:7]} - {commit_subject}")

        # RULE 2: Extract Code Keywords
        try:
            # -C ensures we look at the checked-out bazel source
            changed_files_out = subprocess.check_output(
                ['git', '-C', 'bazel_src', 'show', '--name-only', '--format=', commit_hash],
                text=True
            ).strip()
            changed_files = [f for f in changed_files_out.split('') if f]
        except subprocess.CalledProcessError:
            print(f"⚠️ Could not fetch file list for {commit_hash}")
            continue

        keywords = set()
        for f in changed_files:
            if f.endswith(('.java', '.cc', '.bzl')):
                # Get the class name or filename (e.g., 'AqueryOptions')
                filename = f.split('/')[-1].split('.')[0]
                if len(filename) > 3:
                    keywords.add(filename)

        target_docs = set()

        # RULE 3: GraphQL / GitHub Search Discovery
        # Search up to 5 keywords to find the right documentation
        for kw in list(keywords)[:5]:
            try:
                # 'gh search code' is the most reliable way to map code to docs
                search_out = subprocess.check_output(
                    f"gh search code '{kw}' --repo bazelbuild/bazel --extension mdx --extension md --limit 5",
                    shell=True, text=True, stderr=subprocess.DEVNULL
                )
                
                for line in search_out.strip().split(''):
                    if not line: continue
                    
                    # gh search output is usually "repository:path:content" or "path: content"
                    # We extract the path safely
                    parts = line.split(':')
                    # We look for the first part that looks like a path in docs/ or site/
                    for part in parts:
                        part = part.strip()
                        if ('docs/' in part or 'site/en/' in part) and part.endswith(('.md', '.mdx')):
                            target_docs.add(part)
                            break
            except subprocess.CalledProcessError:
                pass 

        if target_docs:
            print(f"📄 Found matching docs: {target_docs}")
            # RULE 4: Execute Gemini Rewrite
            rewrite_docs_with_gemini(model, commit_subject, note, target_docs)
            
            actionable_commits.append({
                "hash": commit_hash,
                "subject": commit_subject,
                "relnotes": note,
                "docs": sorted(list(target_docs))
            })
        else:
            print("⚠️ No direct documentation match found for this code change. Skipping AI rewrite.")

    if actionable_commits:
        print("
🎉 Documentation rewrite complete for all actionable commits.")
    else:
        print("
😴 No actionable documentation updates found in this run.")

if __name__ == "__main__":
    run_rulebook()
