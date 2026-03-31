 import re
    2 import subprocess
    3 import os
    4 import time
    5 import json
    6 import google.generativeai as genai
    7
    8 def setup_gemini():
    9     api_key = os.environ.get("GEMINI_API_KEY")
   10     if not api_key:
   11         print("Error: GEMINI_API_KEY environment variable not set.")
   12         return None
   13     genai.configure(api_key=api_key)
   14     return genai.GenerativeModel('gemini-2.5-flash'), genai.GenerativeModel('gemini-2.5-pro')
   15
   16 def get_all_doc_paths():
   17     """Gets a list of all current .md and .mdx files in the repo."""
   18     try:
   19         out = subprocess.check_output(
   20             ['git', '-C', 'bazel_src', 'ls-files', 'site/en/**/*.md', 'docs/**/*.mdx'],
   21             text=True
   22         )
   23         paths = [p for p in out.split('\n') if p and '/versions/' not in p and '/archive/' not in
      p]
   24         return paths
   25     except Exception as e:
   26         print(f"Error getting file list: {e}")
   27         return []
   28
   29 def find_best_docs_with_gemini(model, commit_subject, relnote, all_paths):
   30     """PHASE 1: Picks exactly one .md and its corresponding .mdx file."""
   31     paths_str = "\n".join(all_paths)
   32     prompt = f"""
   33     You are an expert Bazel engineer.
   34     Commit: {commit_subject}
   35     Release Note: {relnote}
   36
   37     TASK:
   38     Find the MOST relevant documentation file for this change.
   39     You MUST return EXACTLY TWO files:
   40     1. One DevSite file (starts with 'site/en/' and ends in '.md').
   41     2. One Mintlify file (starts with 'docs/' and ends in '.mdx').
   42     These two files MUST cover the same topic.
   43
   44     Return ONLY a JSON array of strings. Example: ["site/en/ref.md", "docs/ref.mdx"]
   45     If no relevant docs exist, return [].
   46
   47     --- FILES ---
   48     {paths_str}
   49     """
   50     try:
   51         response = model.generate_content(prompt, generation_config={"temperature": 0.0})
   52         raw_json = re.sub(r'^```(?:json)?\s*|\s*

  `$', '', response.text.strip(), flags=re.IGNORECASE)
          paths = json.loads(raw_json)

  --- FIX: FORCE TWIN MAPPING ---
          final_paths = set()
          for p in paths:
              final_paths.add(p)
              if p.startswith('site/en/') and p.endswith('.md'):
  Force add the Mintlify twin
                  twin = p.replace('site/en/', 'docs/').replace('.md', '.mdx')
                  final_paths.add(twin)
              elif p.startswith('docs/') and p.endswith('.mdx'):
  Force add the DevSite twin
                  twin = p.replace('docs/', 'site/en/').replace('.mdx', '.md')
                  final_paths.add(twin)

          return final_paths
  --------------------------------
      except:
          return set()

  def rewrite_docs_with_gemini(model, commit_subject, relnote_text, commit_diff, target_docs):
      """PHASE 2: Surgically updates the file with a strict 4-line limit."""

      for doc_path in target_docs:
          full_path = os.path.join('bazel_src', doc_path)
          if not os.path.exists(full_path): continue

          with open(full_path, 'r', encoding='utf-8') as f:
              original_lines = f.readlines()

          numbered_content = "".join([f"{i+1}: {line}" for i, line in enumerate(original_lines)])

          prompt = f"""
          You are a technical writer. Update this Bazel doc with a MINIMAL note.

          Commit: {commit_subject}
          Note: {relnote_text}
          Diff: {commit_diff[:2000]}

          CRITICAL RULES:
           1. Add or Replace a MAXIMUM of 4 lines. DO NOT write paragraphs.
           2. Be extremely concise. Use one bullet point or two short sentences.
           3. If this is an MDX file, escape braces ({{ -> \\{{).
           4. ALWAYS prefer "insert_after". ONLY use "replace" if the old line is outdated.
           5. Return ONLY JSON:
          {{
              "action": "insert_after" or "replace",
              "line_number": 123,
              "new_text": "Your 1-4 lines of text here"
          }}

          --- DOCUMENT ({doc_path}) ---
          {numbered_content}
          """

          try:
              time.sleep(2)
              response = model.generate_content(prompt, generation_config={"temperature": 0.0})
              raw_json = re.sub(r'^`(?:json)?\s*|\s*

  `$', '', response.text.strip(), flags=re.IGNORECASE)
              update = json.loads(raw_json)

              line_idx = int(update['line_number']) - 1
              new_text = update['new_text'].strip()

  Hard enforcement of 4-line limit in Python
              lines_to_add = new_text.split('\n')
              if len(lines_to_add) > 4:
                  print(f"  ⚠️ AI tried to add {len(lines_to_add)} lines. Truncating to 4.")
                  new_text = "\n".join(lines_to_add[:4])

              if update['action'] == "replace":
                  original_lines[line_idx] = new_text + "\n"
              else:
  insert_after
                  original_lines.insert(line_idx + 1, "\n" + new_text + "\n")

              with open(full_path, 'w', encoding='utf-8') as f:
                  f.writelines(original_lines)

              print(f"  ✅ Surgical Update: {doc_path} ({len(lines_to_add)} lines added/changed)")

          except Exception as e:
              print(f"  ❌ Error for {doc_path}: {e}")

  def run_rulebook():
      flash_model, pro_model = setup_gemini()
      log_path = 'weekly_notes.txt'
      if not os.path.exists(log_path): return

      all_doc_paths = get_all_doc_paths()

      with open(log_path, 'r', encoding='utf-8') as f:
          commits = f.read().split('COMMIT_DELIMITER\n')[1:]

      for commit_block in commits:
          lines = commit_block.strip().split('\n')
          if len(lines) < 3: continue

          commit_hash, commit_subject = lines[0].strip(), lines[1].strip()
          body = '\n'.join(lines[2:])

          match = re.search(r'RELNOTES(?:\[.*?\])?[:\s]+(.*)', body, re.IGNORECASE)
          if not match: continue
          note = match.group(1).strip()
          if re.sub(r'[*`]', '', note).lower().strip() in ['none', 'n/a']: continue

          print(f"\n🚀 Processing: {commit_hash[:7]} - {commit_subject[:50]}...")

  Phase 1: Force find the .md / .mdx PAIR
          target_docs = find_best_docs_with_gemini(flash_model, commit_subject, note, all_doc_paths)

          if target_docs:
              print(f"  🎯 Target Pair: {list(target_docs)}")
              try:
                  diff = subprocess.check_output(['git', '-C', 'bazel_src', 'show', '--format=',
  commit_hash], text=True).strip()
  Phase 2: Perform minimal edit on BOTH files in the pair
                  rewrite_docs_with_gemini(pro_model, commit_subject, note, diff, target_docs)
              except Exception as e:
                  print(f"  ⚠️ Error: {e}")
          else:
              print("  ⏭️ No matching documentation pair found.")

  if __name__ == "__main__":
