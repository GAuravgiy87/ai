
import subprocess
import os

commits = [
    "2ea8f0ac0fddf6233101bd8f62c5b7f49f2e38d0",
    "d7c82fa0e34b42515983790d68ae8d56605656d9",
    "cc42a3551b1299fd73e50333fa11bc88e0385d9c",
    "a5ea062367072391ecfd5a8f4cd411ad209d5a44",
    "09e015cb8bbf6ca800fd453025a6bb1e9c44dbdd",
    "21a870d1f4c14b2e3fdbbf60eb29c00417906c24",
    "00eb3370227571e019cb9f8f4b5f416834829cdd",
    "cdfb433d61924271dbfeafef07d1e46c179e9274",
    "33b1588d17ea044255ae2d50824187b6c4e81804"
]

output_file = "COMMIT_DIFFS.md"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("# AI Vigilance: Full Commit Diffs\n")
    f.write("---\n\n")
    f.write("## Full Git Diffs for All Commits\n\n")
    f.write("---\n\n")

    for commit in commits:
        print(f"Processing {commit}...")
        result = subprocess.run(
            ["git", "show", commit],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace"
        )
        
        f.write(f"\n---\n\n")
        f.write(f"## Commit: `{commit}`\n")
        f.write("\n```diff\n")
        f.write(result.stdout)
        f.write("\n```\n")

print(f"All diffs saved to {output_file}!")
