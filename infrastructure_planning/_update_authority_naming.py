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

# Replacement for Rule 30 Naming Convention Table
new_naming_table = """| Scope | Usage | Example |
|-------|-------|---------|
| ceo/ | Strategic Development & Rule Management (CEO Only) | ceo/kuro-semantic-event-structures |
| infra/ | Infrastructure / DevOps / MLOps | infra/milestone-0-setup |
| feat/ | New feature development | feat/MLO-1-ci-cd-pipeline |
| fix/ | Bug fix | fix/MLO-3-docker-volume-error |
| docs/ | Documentation only | docs/update-readme-badges |
| refactor/ | Code refactoring | refactor/modularize-training |

5. **Global Consistency**: For tasks that span multiple repositories (e.g., rule syncs, platform migrations), the branch name MUST be identical across all affected repositories."""

# Replacement for Rule 33 Authority
new_rule33_content = """## RULE 33: Global Rule Parity and Mandatory Cross-Branch Sync -- CRITICAL

### Rule
The AI rule set (AGENTS.md, AI_GUIDELINES.md, .cursorrules) represents the immutable "Physical Laws" of the repository ecosystem. Rules are **global** and MUST NOT vary between branches. 

### Authority Restriction
Only branches with the **`ceo/`** scope have the authority to modify rule files. Any rule changes attempted on `infra/`, `feat/`, or other branches MUST be rejected by the AI Agent. Non-CEO branches MUST merge rule updates FROM a `ceo/` branch to maintain parity.

### Mandatory Sync Process
1. **Rule Modification**: When any rule is added or modified on a `ceo/` branch, the AI Agent MUST immediately:
   - Commit the change on the current branch.
   - Switch to all other active development branches (e.g., `infra/milestone-0-setup`, `main`) and merge the changes.
   - Update the master `kuro-rules` repository.
2. **Review Enforcement**: No Pull Request (PR) can be merged without explicitly confirming that the branch has the status of the "Current Rule Set" (Rule 33 verification)."""

for f in files:
    try:
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        
        # Update Rule 30 Table
        if '| Scope | Usage | Example |' in content:
            # Finding the end of the table to replace it
            lines = content.split('\n')
            start_idx = -1
            end_idx = -1
            for i, line in enumerate(lines):
                if '| Scope | Usage | Example |' in line:
                    start_idx = i
                if start_idx != -1 and line.strip() == '' and i > start_idx + 2:
                    end_idx = i
                    break
            if start_idx != -1 and end_idx != -1:
                lines[start_idx:end_idx] = [new_naming_table]
                content = '\n'.join(lines)

        # Update Rule 33 Content
        if '## RULE 33:' in content:
            # Replace the whole section from ## RULE 33 until the end or next ##
            import re
            content = re.sub(r'## RULE 33:.*?(?=---|$)', new_rule33_content + '\n\n', content, flags=re.DOTALL)

        with open(f, 'w', encoding='utf-8') as out_file:
            out_file.write(content)
    except Exception as e:
        print(f'Failed {f}: {e}')

print('Updated Rule 30 and 33 in all files.')
