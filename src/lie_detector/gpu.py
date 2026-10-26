import glob
import os
import sys

_done = False


def _dll_dirs() -> list[str]:
    if os.environ.get("ORT_DLL_DIRS"):
        return os.environ["ORT_DLL_DIRS"].split(os.pathsep)
    pf = os.environ.get("ProgramFiles", r"C:\Program Files")
    g = lambda *p: sorted(glob.glob(os.path.join(pf, *p)), reverse=True)
    return g("NVIDIA GPU Computing Toolkit", "CUDA", "v13*", "bin", "x64")[:1] + \
        (g("NVIDIA", "CUDNN", "v9*", "bin", "13*", "x64") + g("NVIDIA", "CUDNN", "v9*", "bin", "13*"))[:1]


def setup_onnxruntime() -> list[str]:
    global _done
    import onnxruntime as ort

    ort.set_default_logger_severity(3)
    if sys.platform == "win32" and not _done:
        for d in filter(os.path.isdir, _dll_dirs()):
            os.add_dll_directory(d)
            os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]
        _done = True
    return ort.get_available_providers()
