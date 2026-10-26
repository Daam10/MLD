import argparse

from lie_detector.config import sessions


def session_args(*flags):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", nargs="*")
    ap.add_argument("--force", action="store_true")
    for f in flags:
        ap.add_argument(f, action="store_true")
    args = ap.parse_args()
    return args, {sid: s for sid, s in sessions().items() if not args.sessions or sid in args.sessions}
