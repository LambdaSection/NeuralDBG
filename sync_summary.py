import os
import sys
import subprocess

def check_pandoc():
    try:
        subprocess.run(["pandoc", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def convert_summary_to_docx(target_dir):
    summary_path = os.path.join(target_dir, "SESSION_SUMMARY.md")
    output_path = os.path.join(target_dir, "SESSION_SUMMARY.docx")
    
    if not os.path.exists(summary_path):
        print(f"[-] Fichier introuvable : {summary_path}")
        return False
        
    if not check_pandoc():
        print("[-] ERREUR: Pandoc n'est pas installé ou n'est pas dans le PATH.")
        print("    Veuillez l'installer via : winget install pandoc")
        return False
        
    try:
        print(f"[*] Conversion en cours : {summary_path} -> {output_path}")
        subprocess.run(["pandoc", summary_path, "-o", output_path], check=True)
        print(f"[+] Conversion réussie : {output_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[-] Erreur lors de la conversion : {e}")
        return False

if __name__ == "__main__":
    # Par défaut, on regarde dans le dossier courant
    target_directory = "."
    
    if len(sys.argv) > 1:
        target_directory = sys.argv[1]
        
    convert_summary_to_docx(target_directory)
