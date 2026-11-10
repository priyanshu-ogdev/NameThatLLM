import ast

def audit_file(filepath):
    print(f"Auditing {filepath} for undefined variables in del statements and scope issues...")
    with open(filepath, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read())

    issues = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined_names = set()
            for arg in node.args.args:
                defined_names.add(arg.arg)
            if node.args.vararg:
                defined_names.add(node.args.vararg.arg)
            if node.args.kwarg:
                defined_names.add(node.args.kwarg.arg)

            for child in ast.walk(node):
                if isinstance(child, (ast.Assign, ast.AnnAssign)):
                    targets = child.targets if isinstance(child, ast.Assign) else [child.target]
                    for target in targets:
                        for subtarget in ast.walk(target):
                            if isinstance(subtarget, ast.Name):
                                defined_names.add(subtarget.id)
                elif isinstance(child, (ast.For, ast.AsyncFor)):
                    for subtarget in ast.walk(child.target):
                        if isinstance(subtarget, ast.Name):
                            defined_names.add(subtarget.id)
                elif isinstance(child, (ast.Import, ast.ImportFrom)):
                    for alias in child.names:
                        defined_names.add(alias.asname or alias.name)
                elif isinstance(child, ast.FunctionDef):
                    defined_names.add(child.name)
                elif isinstance(child, ast.With):
                    for item in child.items:
                        if item.optional_vars:
                            for subtarget in ast.walk(item.optional_vars):
                                if isinstance(subtarget, ast.Name):
                                    defined_names.add(subtarget.id)
                elif isinstance(child, ast.ExceptHandler):
                    if child.name:
                        defined_names.add(child.name)

            for child in ast.walk(node):
                if isinstance(child, ast.Delete):
                    for target in child.targets:
                        if isinstance(target, ast.Name):
                            if target.id not in defined_names:
                                issues.append((node.name, target.lineno, target.id))

    if issues:
        print(f"FOUND {len(issues)} ISSUES:")
        for func_name, lineno, var_name in issues:
            print(f"  Line {lineno} in {func_name}: del {var_name} (NOT DEFINED IN FUNCTION SCOPE)")
    else:
        print("No undefined del statements found.")

if __name__ == "__main__":
    audit_file("notebooks/kaggle_tri_ensemble.py")
    audit_file("notebooks/kaggle_dry_run.py")

