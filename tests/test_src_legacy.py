# tests/test_no_src_imports.py
from pathlib import Path

# check that roseate is not dependent on legacy code in /src/
def test_roseate_does_not_import_src():
   for f in Path("roseate").glob("*.py"):
       text = f.read_text()
       assert "from src" not in text and "import src" not in text, f

       