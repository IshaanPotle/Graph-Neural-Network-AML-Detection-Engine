import os
import sys

REQUIRED_FILES = [
    "data/elliptic_raw/wallets_features.csv",
    "data/elliptic_raw/AddrAddr_edgelist.csv",
    "data/elliptic_raw/wallets_classes.csv"
]

REQUIRED_PACKAGES = [
    "torch", "torch_geometric", "dgl", "pandas", "numpy", "sklearn",
    "fastapi", "uvicorn", "streamlit", "networkx", "plotly", "pyvis", "loguru"
]

REQUIRED_IMPORTS = [
    "torch",
    "torch_geometric",
    "dgl",
    "pandas",
    "numpy",
    "sklearn",
    "networkx",
    "plotly",
    "pyvis",
    "loguru",
    "fastapi",
    "uvicorn",
    "streamlit"
]

def check_files():
    print("Checking required data files...")
    missing = []
    for f in REQUIRED_FILES:
        if not os.path.exists(f):
            print(f"❌ MISSING: {f}")
            missing.append(f)
        else:
            print(f"✅ Found: {f}")
    return missing

def check_packages():
    print("\nChecking required Python packages...")
    import importlib
    missing = []
    for pkg in REQUIRED_PACKAGES:
        try:
            # Handle special case for sklearn
            if pkg == "sklearn":
                importlib.import_module("sklearn")
            else:
                importlib.import_module(pkg.replace("-", "_"))
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ MISSING: {pkg}")
            missing.append(pkg)
    return missing

def check_imports():
    print("\nChecking key imports...")
    failed = []
    for imp in REQUIRED_IMPORTS:
        try:
            __import__(imp)
            print(f"✅ import {imp}")
        except Exception as e:
            print(f"❌ import {imp} failed: {e}")
            failed.append(imp)
    # Check project-specific imports
    try:
        from data.loader import GNNInputLoader, validate_graph_data
        from models.graphsage import GraphSAGE
        from inference.inference import AMLInferenceEngine
        print("✅ Project-specific imports")
    except Exception as e:
        print(f"❌ Project-specific imports failed: {e}")
        failed.append("project-specific")
    return failed

def check_loader():
    print("\nChecking data loader...")
    try:
        from data.loader import GNNInputLoader, validate_graph_data
        loader = GNNInputLoader(framework='pyg', data_path='./data')
        graph_data = loader.load_elliptic_dataset(
            nodes_file='wallets_features.csv',
            edges_file='AddrAddr_edgelist.csv',
            classes_file='wallets_classes.csv'
        )
        assert validate_graph_data(graph_data)
        print("✅ Data loaded and validated!")
        return True
    except Exception as e:
        print(f"❌ Data loader failed: {e}")
        return False

if __name__ == "__main__":
    print("=== AML Engine Environment Auto-Check ===\n")
    files_missing = check_files()
    pkgs_missing = check_packages()
    imports_failed = check_imports()
    loader_ok = check_loader()

    print("\n=== SUMMARY ===")
    if not files_missing and not pkgs_missing and not imports_failed and loader_ok:
        print("🎉 All checks passed! You are ready to run the AML Engine.")
    else:
        print("⚠️ Please fix the above issues before running the pipeline.") 