"""``picasso-workflow-admin-cli`` -- headless front-end over :mod:`.core`.

Same operations as the GUI (``images`` / ``jobs`` / ``test``) for scripting or a
quick look without a display. Connection settings mirror the Run tab:
``--host`` is a cluster key from ``SlurmLoginNodes`` (or a bare hostname),
``--user`` defaults to the current shell user, ``--identity`` to the discovered
SSH key.
"""

from __future__ import annotations

import argparse
import sys

from picasso_workflow.admin import core


def _build_admin(args):
    return core.build_admin(
        args.host, username=args.user, ssh_key_path=args.identity
    )


def _cmd_test(args):
    admin = _build_admin(args)
    ok = admin.test_connection()
    print("ok" if ok else "unreachable")
    return 0 if ok else 1


def _cmd_images(args):
    admin = _build_admin(args)
    images = admin.list_images()
    gui_version = core.this_gui_version()
    for im in images:
        match = " *this-version" if im.version == gui_version else ""
        print(
            f"{im.name}\t{im.version or '-'}\t"
            f"{'test' if im.is_test else 'lean'}\t{im.size}\t{im.mtime}{match}"
        )
    return 0


def _cmd_jobs(args):
    admin = _build_admin(args)
    disk = admin.disk_usage()
    if disk is not None:
        print(
            f"disk {disk.path}: {disk.used}/{disk.size} "
            f"({disk.use_pct}), {disk.avail} free"
        )
    print("== queue ==")
    for j in admin.queue():
        print(f"{j.job_id}\t{j.name}\t{j.state}\t{j.elapsed}\t{j.extra}")
    print("== recent ==")
    for j in admin.recent_jobs():
        print(f"{j.job_id}\t{j.name}\t{j.state}\t{j.elapsed}\t{j.extra}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="picasso-workflow-admin-cli",
        description="Inspect cluster Apptainer images, the SLURM queue, and "
        "disk usage over SSH.",
    )
    parser.add_argument(
        "--host",
        required=True,
        help="cluster key (SlurmLoginNodes) or a login-node hostname",
    )
    parser.add_argument(
        "--user", default=None, help="SSH user (default $USER)"
    )
    parser.add_argument(
        "--identity", default=None, help="SSH private key path"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("test", help="test the SSH connection")
    sub.add_parser("images", help="list .sif images on the shared mount")
    sub.add_parser("jobs", help="show the SLURM queue, recent jobs, disk use")

    args = parser.parse_args(argv)
    handler = {
        "test": _cmd_test,
        "images": _cmd_images,
        "jobs": _cmd_jobs,
    }[args.command]
    try:
        return handler(args)
    except Exception as exc:  # friendly one-line error, non-zero exit
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
