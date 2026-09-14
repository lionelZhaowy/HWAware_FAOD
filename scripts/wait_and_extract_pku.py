"""Wait for DSEC migration, then resume PKU ZIP extraction safely.

Runs as a standalone background process. Does not move/delete DSEC or PEOD.
Existing files are checked against ZIP size/CRC; replacements are atomic.
"""
import fcntl
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import time
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / 'logs' / 'pku_auto_extract'
ARCHIVE = Path('/data/lab_dataset/RGB_DVS_DET/FAOD_PKU_DAVIS_SOD.zip')
DEST = Path('/home/zhaowenyao24/Conda_prj/lab_dataset/FAOD_PKU_DAVIS_SOD')
OLD = Path('/srv/datasets/DSEC_DET')
MOVED = Path('/data/lab_dataset/RGB_DVS_DET/DSEC_DET')
RESERVE = 20 * 1024**3


def status(state, **extra):
    record = dict(state=state, pid=os.getpid(), updated_at=time.strftime('%Y-%m-%d %H:%M:%S %z'), **extra)
    tmp = CONTROL / 'status.tmp'
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(CONTROL / 'status.json')
    print(json.dumps(record, ensure_ascii=False), flush=True)


def valid_file(path, member):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != member.file_size:
        return False
    checksum = 0
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            checksum = zlib.crc32(block, checksum)
    return checksum & 0xffffffff == member.CRC


def other_extractors():
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        if proc.name == str(os.getpid()):
            continue
        try:
            args = (proc / 'cmdline').read_bytes().split(b'\0')
        except (OSError, PermissionError):
            continue
        # Detect the previously used Python CLI and other copies of this worker.
        if (b'zipfile' in args and str(ARCHIVE).encode() in args) or any(
            arg.endswith(b'/wait_and_extract_pku.py') for arg in args
        ):
            found.append(int(proc.name))
    return found


def main():
    CONTROL.mkdir(parents=True, exist_ok=True)
    with (CONTROL / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (CONTROL / 'worker.pid').write_text(str(os.getpid()) + '\n')
        while not (OLD.is_symlink() and OLD.resolve() == MOVED and MOVED.is_dir()):
            status('waiting_for_dsec_migration', required_link=str(OLD), target=str(MOVED))
            time.sleep(60)
        with zipfile.ZipFile(ARCHIVE) as archive:
            members = archive.infolist()
            for member in members:
                rel = PurePosixPath(member.filename)
                if rel.is_absolute() or '..' in rel.parts or not rel.parts or rel.parts[0] != 'freq_1_1':
                    raise ValueError(f'Unexpected archive path: {member.filename}')
            total = sum(m.file_size for m in members)
            # Conservative: require room for the entire archive plus headroom,
            # even though a fraction of its files may already exist.
            while True:
                free = shutil.disk_usage(DEST.parent).free
                others = other_extractors()
                if free >= total + RESERVE and not others:
                    break
                status('waiting_for_space_or_extractor', free_bytes=free,
                       required_bytes=total + RESERVE, other_pids=others)
                time.sleep(60)
            DEST.mkdir(parents=True, exist_ok=True)
            skipped = written = 0
            for index, member in enumerate(members):
                target = DEST / member.filename
                # Refuse existing symlinks anywhere in the output path.
                if any(p.is_symlink() for p in [target, *target.parents] if p != DEST.parent):
                    raise ValueError(f'Symlink in output path: {target}')
                if member.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                if valid_file(target, member):
                    skipped += 1
                else:
                    while shutil.disk_usage(DEST).free < member.file_size + RESERVE:
                        status('waiting_for_space', member=member.filename, reserve_bytes=RESERVE)
                        time.sleep(60)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temporary = target.with_name(target.name + '.faod-extract-part')
                    checksum = count = 0
                    with archive.open(member) as source, temporary.open('wb') as output:
                        for block in iter(lambda: source.read(4 * 1024**2), b''):
                            output.write(block)
                            checksum = zlib.crc32(block, checksum)
                            count += len(block)
                        output.flush()
                        os.fsync(output.fileno())
                    if count != member.file_size or checksum & 0xffffffff != member.CRC:
                        raise ValueError(f'CRC/size mismatch: {member.filename}')
                    temporary.replace(target)
                    written += 1
                status('extracting', archive_index=index + 1, archive_entries=len(members),
                       member=member.filename, skipped_verified=skipped, written_verified=written)
            status('complete', dataset_root=str(DEST / 'freq_1_1'),
                   skipped_verified=skipped, written_verified=written, archive_bytes=ARCHIVE.stat().st_size)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if CONTROL.exists():
            # Do not overwrite the active worker's status if a duplicate launch fails.
            if not isinstance(exc, BlockingIOError):
                status('failed', error=repr(exc))
        raise
