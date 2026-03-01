import os

files = [
    r'c:\Users\Utilisateur\Documents\NeuralDBG\AGENTS.md',
    r'c:\Users\Utilisateur\Documents\NeuralDBG\AI_GUIDELINES.md',
    r'c:\Users\Utilisateur\Documents\NeuralDBG\.cursorrules',
    r'c:\Users\Utilisateur\Documents\NeuralDBG\copilot-instructions.md',
    r'c:\Users\Utilisateur\Documents\NeuralDBG\GAD.md',
    r'c:\Users\Utilisateur\Documents\kuro-rules\AGENTS.md',
    r'c:\Users\Utilisateur\Documents\kuro-rules\AI_GUIDELINES.md',
    r'c:\Users\Utilisateur\Documents\kuro-rules\.cursorrules',
    r'c:\Users\Utilisateur\Documents\kuro-rules\copilot-instructions.md',
    r'c:\Users\Utilisateur\Documents\kuro-rules\GAD.md'
]

rule33 = """

---

## RULE 33: Global Rule Parity and Mandatory Cross-Branch Sync -- CRITICAL

### Rule
The AI rule set (AGENTS.md, AI_GUIDELINES.md, .cursorrules) represents the immutable "Physical Laws" of the repository ecosystem. Rules are **global** and MUST NOT vary between branches. 

### Mandatory Sync Process
1. **Rule Modification**: When any rule is added or modified on a development branch, the AI Agent MUST immediately:
   - Commit the change on the current branch.
   - Switch to all other active development branches (e.g., `infra/milestone-0-setup`, `main`) and merge the changes.
   - Update the master `kuro-rules` repository.
2. **Review Enforcement**: No Pull Request (PR) can be merged without explicitly confirming that the branch has the status of the "Current Rule Set" (Rule 33 verification).

### Enforcement
```
IF a rule is changed:
  ACTION: SYNC (merge) the modified rule files to ALL active branches BEFORE proceeding with code.
  ACTION: Update SYNC_LOG.md and SESSION_SUMMARY.md.
  DO NOT: Allow branch-specific rules.
  DO NOT: Work on an outdated rule set.
```
"""

count = 0
for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        if 'RULE 33' not in content:
            content = content.rstrip() + rule33
            with open(f, 'w', encoding='utf-8') as out_file:
                out_file.write(content)
            count += 1
    except Exception as e:
        print(f'Failed {f}: {e}')

print(f'Added Rule 33 to {count}/10 rule files.')
