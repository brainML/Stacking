import re
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
HOME_PATH = re.compile(r"/(?:Users|home)/[^/\s]+/")
EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)


class PublicTreePrivacyTests(unittest.TestCase):
    def test_public_text_has_no_home_paths_or_email_addresses(self):
        public_files = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        ).stdout.decode().split("\0")
        violations = []
        for relative in public_files:
            path = ROOT / relative
            if not relative or path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            text = path.read_text(encoding="utf-8")
            if HOME_PATH.search(text) or EMAIL.search(text):
                violations.append(relative)
        self.assertEqual(violations, [], f"private identifiers found in {violations}")


if __name__ == "__main__":
    unittest.main()
