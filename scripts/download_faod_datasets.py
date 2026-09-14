#!/usr/bin/env python3
"""Download the three public FAOD archives with anonymous cookies and resume.

No Microsoft login is needed. Existing dataset directories are not modified.
Run inside tmux or with nohup; status and logs are kept in DEST/.faod_download.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import time
import zipfile

import requests


DATASETS = [
    ('PKU-DAVIS-SOD', 'FAOD_PKU_DAVIS_SOD.zip', 54486709692),
    ('DSEC-Detection', 'FAOD_DSEC_Detection.zip', 86801382924),
    ('EOD200', 'FAOD_EOD200.zip', 47797567175),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    control = destination / '.faod_download'
    control.mkdir(exist_ok=True, mode=0o700)
    lock = (control / 'lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    root = Path(__file__).resolve().parents[1]
    urls = re.findall(r'<a href="(https://entuedu-my\.sharepoint\.com/[^\"]+)"',
                      (root / 'README.md').read_text())[:3]
    if len(urls) != 3:
        raise RuntimeError('Expected three FAOD dataset URLs in README')
    manifest = [dict(name=name, filename=filename, bytes=size,
                     url=url + ('&' if '?' in url else '?') + 'download=1')
                for (name, filename, size), url in zip(DATASETS, urls)]
    (control / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    state = {'pid': os.getpid(), 'started_at': time.strftime('%Y-%m-%d %H:%M:%S %z'),
             'state': 'starting', 'datasets': {}}

    def update(name=None, **values):
        if name is not None:
            state['datasets'].setdefault(name, {}).update(values)
        else:
            state.update(values)
        state['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S %z')
        temp = control / 'status.json.tmp'
        temp.write_text(json.dumps(state, indent=2) + '\n')
        temp.replace(control / 'status.json')

    def log(message):
        print(time.strftime('%Y-%m-%d %H:%M:%S'), message, flush=True)

    # Allow for archives plus an extra 10 GiB. Extraction is a separate operation.
    remaining = 0
    for entry in manifest:
        final = destination / entry['filename']
        part = destination / (entry['filename'] + '.part')
        existing = final if final.exists() else part
        present = existing.stat().st_size if existing.exists() else 0
        remaining += max(0, entry['bytes'] - present)
        update(entry['name'], state='queued', bytes=present, total_bytes=entry['bytes'])
    if shutil.disk_usage(destination).free < remaining + 10 * 2**30:
        update(state='failed', error='Insufficient free space for archives plus 10 GiB margin')
        raise RuntimeError(state['error'])
    update(state='running')
    for entry in manifest:
        name, size = entry['name'], entry['bytes']
        final = destination / entry['filename']
        part = destination / (entry['filename'] + '.part')
        if final.exists():
            if final.stat().st_size != size:
                raise RuntimeError(f'Existing file has unexpected size; refusing to overwrite {final.name}')
            target = final
        else:
            failures = 0
            while not part.exists() or part.stat().st_size < size:
                offset = part.stat().st_size if part.exists() else 0
                checkpoint_time, checkpoint_bytes = time.monotonic(), offset
                update(name, state='downloading', bytes=offset, total_bytes=size)
                log(f'{name}: requesting bytes {offset}- of {size}')
                try:
                    # New anonymous session refreshes sharing cookies on every retry.
                    with requests.Session() as session:
                        with session.get(entry['url'], headers={'Range': f'bytes={offset}-',
                                  'Accept-Encoding': 'identity'}, stream=True, timeout=(20, 90)) as response:
                            if response.status_code != 206:
                                raise RuntimeError(f'Expected HTTP 206, received {response.status_code}')
                            expected = f'bytes {offset}-{size-1}/{size}'
                            if response.headers.get('Content-Range') != expected:
                                raise RuntimeError('Unexpected Content-Range; refusing to append')
                            if 'zip' not in response.headers.get('Content-Type', '').lower():
                                raise RuntimeError('Response is not a ZIP file')
                            with part.open('ab') as output:
                                for chunk in response.iter_content(chunk_size=1024 * 1024):
                                    if not chunk:
                                        continue
                                    if offset + len(chunk) > size:
                                        raise RuntimeError('Response exceeds expected file size')
                                    output.write(chunk)
                                    offset += len(chunk)
                                    now = time.monotonic()
                                    if now - checkpoint_time >= 30:
                                        output.flush()
                                        speed = (offset - checkpoint_bytes) / (now - checkpoint_time)
                                        update(name, bytes=offset, bytes_per_second=round(speed),
                                               percent=round(100 * offset / size, 2))
                                        log(f'{name}: {offset}/{size} bytes ({100*offset/size:.2f}%), {speed/2**20:.2f} MiB/s')
                                        checkpoint_time, checkpoint_bytes = now, offset
                                output.flush()
                                os.fsync(output.fileno())
                            if offset != size:
                                raise RuntimeError('Stream ended before expected size')
                    failures = 0
                except (requests.RequestException, RuntimeError) as error:
                    failures += 1
                    # Do not log signed redirect URLs, cookies or request headers.
                    kind = type(error).__name__
                    message = str(error) if isinstance(error, RuntimeError) else kind
                    update(name, state='retrying', bytes=part.stat().st_size if part.exists() else 0,
                           last_error=message, consecutive_failures=failures)
                    log(f'{name}: {message}; resume retry {failures}/30')
                    if failures >= 30:
                        update(state='failed')
                        raise RuntimeError(f'{name}: retry budget exhausted') from None
                    time.sleep(min(10 * failures, 120))
            target = part
        if target.stat().st_size != size:
            raise RuntimeError(f'{name}: unexpected completed file size')
        update(name, state='verifying_crc', bytes=size, percent=100)
        log(f'{name}: download complete; checking ZIP structure and all file CRCs')
        with zipfile.ZipFile(target) as archive:
            members = archive.infolist()
            update(name, zip_members=len(members), uncompressed_bytes=sum(x.file_size for x in members))
            bad_file = archive.testzip()
            if bad_file is not None:
                raise RuntimeError(f'{name}: CRC failure; retaining archive for inspection')
        if target == part:
            part.rename(final)
        update(name, state='complete', path=str(final), crc_verified=True)
        log(f'{name}: verified and saved as {final}')
    update(state='complete')
    log('All three datasets downloaded and ZIP CRC checks passed.')


if __name__ == '__main__':
    main()
