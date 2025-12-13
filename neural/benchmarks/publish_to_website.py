#!/usr/bin/env python
"""
Script to publish benchmark results to a website or GitHub Pages.
"""

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path


def publish_to_github_pages(report_dir: Path, gh_pages_dir: Path):
    print(f"Publishing to GitHub Pages: {gh_pages_dir}")
    
    benchmarks_dir = gh_pages_dir / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)
    
    latest_dir = benchmarks_dir / "latest"
    if latest_dir.exists():
        archive_name = f"archived_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        archive_dir = benchmarks_dir / archive_name
        shutil.move(str(latest_dir), str(archive_dir))
        print(f"  Archived previous results to {archive_name}")
    
    shutil.copytree(str(report_dir), str(latest_dir))
    print(f"  ✓ Copied results to {latest_dir}")
    
    index_html = gh_pages_dir / "index.html"
    if not index_html.exists():
        template_path = Path(__file__).parent / "website_template.html"
        if template_path.exists():
            shutil.copy(str(template_path), str(index_html))
            print(f"  ✓ Created index.html")
    
    readme_md = gh_pages_dir / "README.md"
    readme_content = f"""# Neural DSL Benchmark Results

Official benchmark results comparing Neural DSL against popular ML frameworks.

## Latest Results

View the interactive report: [benchmarks/latest/index.html](benchmarks/latest/index.html)

## Benchmark Archive

"""
    
    for archived in sorted(benchmarks_dir.glob("archived_*"), reverse=True):
        readme_content += f"- [{archived.name}](benchmarks/{archived.name}/index.html)\n"
    
    readme_content += f"""
## Reproduction

To reproduce these benchmarks:

```bash
git clone https://github.com/Lemniscate-world/Neural.git
cd Neural
pip install -e ".[full]"
python neural/benchmarks/run_benchmarks.py
```

Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    with open(readme_md, "w") as f:
        f.write(readme_content)
    
    print(f"  ✓ Updated README.md")
    print("\n✓ Publishing complete!")
    print(f"\nNext steps:")
    print(f"  1. cd {gh_pages_dir}")
    print(f"  2. git add .")
    print(f"  3. git commit -m 'Update benchmark results'")
    print(f"  4. git push")


def publish_to_directory(report_dir: Path, output_dir: Path):
    print(f"Publishing to directory: {output_dir}")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest_dir = output_dir / f"benchmark_{timestamp}"
    
    shutil.copytree(str(report_dir), str(dest_dir))
    
    latest_link = output_dir / "latest"
    if latest_link.exists():
        if latest_link.is_symlink():
            latest_link.unlink()
        else:
            shutil.rmtree(latest_link)
    
    try:
        latest_link.symlink_to(dest_dir.name)
        print(f"  ✓ Created symlink: latest -> {dest_dir.name}")
    except OSError:
        shutil.copytree(str(dest_dir), str(latest_link))
        print(f"  ✓ Copied to latest directory")
    
    print(f"  ✓ Published to {dest_dir}")
    print("\n✓ Publishing complete!")


def main():
    parser = argparse.ArgumentParser(
        description="Publish benchmark results to website or GitHub Pages"
    )
    parser.add_argument(
        "report_dir",
        type=str,
        help="Path to benchmark report directory",
    )
    parser.add_argument(
        "--github-pages",
        type=str,
        help="Path to GitHub Pages repository (e.g., docs/)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory for publishing",
    )
    
    args = parser.parse_args()
    
    report_dir = Path(args.report_dir)
    if not report_dir.exists():
        print(f"✗ Report directory not found: {report_dir}")
        return 1
    
    print("=" * 60)
    print("Neural DSL Benchmark Publisher")
    print("=" * 60)
    
    if args.github_pages:
        gh_pages_dir = Path(args.github_pages)
        publish_to_github_pages(report_dir, gh_pages_dir)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        publish_to_directory(report_dir, output_dir)
    else:
        print("✗ Must specify either --github-pages or --output-dir")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
