import os

# Set environment variables before juliacall/juliapkg is imported.
# These ensure Julia 1.12 (the system-installed Julia) is used in offline mode,
# avoiding the need to download or resolve Julia packages at test time.
if "PYTHON_JULIAPKG_EXE" not in os.environ:
    import shutil
    julia_exe = shutil.which("julia")
    if julia_exe:
        os.environ["PYTHON_JULIAPKG_EXE"] = julia_exe

if "PYTHON_JULIAPKG_OFFLINE" not in os.environ:
    os.environ["PYTHON_JULIAPKG_OFFLINE"] = "yes"
