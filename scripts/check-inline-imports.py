#!/usr/bin/env python3
"""Check for inline imports that should be moved to the top of the file."""

import argparse
import ast
import sys
from pathlib import Path


class InlineImportVisitor(ast.NodeVisitor):
    """AST visitor to find imports that are not at module level."""

    def __init__(self):
        self.inline_imports: list[tuple[int, str, str]] = []
        self.function_depth = 0
        self.class_depth = 0
        self.in_conditional = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_depth += 1
        self.generic_visit(node)
        self.class_depth -= 1

    def visit_If(self, node: ast.If) -> None:
        old_conditional = self.in_conditional
        self.in_conditional = True
        self.generic_visit(node)
        self.in_conditional = old_conditional

    def visit_Try(self, node: ast.Try) -> None:
        old_conditional = self.in_conditional
        self.in_conditional = True
        self.generic_visit(node)
        self.in_conditional = old_conditional

    def visit_Import(self, node: ast.Import) -> None:
        if self._is_inline_import():
            names = ", ".join(alias.name for alias in node.names)
            self.inline_imports.append((node.lineno, "import", names))
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if self._is_inline_import():
            module = node.module or ""
            names = ", ".join(alias.name for alias in node.names)
            import_text = (
                f"from {module} import {names}" if module else f"import {names}"
            )
            self.inline_imports.append((node.lineno, "from", import_text))
        self.generic_visit(node)

    def _is_inline_import(self) -> bool:
        """Check if current import is inline (inside function, class, or conditional)."""
        return self.function_depth > 0 or self.class_depth > 0 or self.in_conditional


def check_file(filepath: Path) -> list[tuple[int, str, str]]:
    """Check a single Python file for inline imports."""
    try:
        content = filepath.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(filepath))

        visitor = InlineImportVisitor()
        visitor.visit(tree)

        return visitor.inline_imports
    except SyntaxError as e:
        print(f"Syntax error in {filepath}: {e}")
        return []
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return []


def main() -> int:
    """Main function to check for inline imports."""
    parser = argparse.ArgumentParser(description="Check for inline imports")
    parser.add_argument("files", nargs="*", help="Python files to check")
    parser.add_argument(
        "--check-all",
        action="store_true",
        help="Check all Python files in src/ directory",
    )

    args = parser.parse_args()

    if args.check_all:
        files = list(Path("src").rglob("*.py"))
    else:
        files = [Path(f) for f in args.files if f.endswith(".py")]

    total_issues = 0

    for filepath in files:
        if not filepath.exists():
            continue

        inline_imports = check_file(filepath)

        if inline_imports:
            print(f"\n{filepath}:")
            for lineno, _, import_text in inline_imports:
                print(f"  Line {lineno}: {import_text}")
                total_issues += 1

    if total_issues > 0:
        print(
            f"\nFound {total_issues} inline import(s) that might be moved to the top of the file."
        )
        print(
            "Note: Some inline imports are intentional (conditional imports, avoiding circular imports)."
        )
        print("Review each case to determine if it should be moved to the top.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
