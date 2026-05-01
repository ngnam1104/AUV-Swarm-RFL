import re
import os
filepath = r'd:\Documents\HUST\2022-2026\Research_Thesis\AUV-Swarm-RFL\Research_Proposal.md'
if not os.path.exists(filepath):
    print(f"File not found: {filepath}")
    exit(1)
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()
# Pattern to match [cite: ...] potentially with a leading space
# We use \s? to include optional leading space
cleaned_content = re.sub(r'\s?\[cite: [^\]]+\]', '', content)
with open(filepath, 'w', encoding='utf-8') as f:
    f.write(cleaned_content)
print("Successfully removed all [cite: ...] occurrences.")