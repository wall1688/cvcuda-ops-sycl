#!/usr/bin/env python3
# Transfer the SYCL NMS port from the local staging tree to the Intel board's
# /work/bevfusion_migration/ via the pexpect SSH tunnel
# (local -> jump host -> ./connect_<intel>.sh -> Intel host), mirroring the
# colleague's scripts/ssh_transfer_build.py.
#
# scp jump->intel failed in the original workflow, so we stream base64 in
# <1500-char chunks through the PTY and verify each file with md5sum.
#
# Credentials are read from env vars (fall back to redacted placeholders):
#   JUMP_PW     password for the jump host
#   JUMP_HOST   user@host of the jump host
#   CONNECT_SH  the connect script on the jump host that reaches the Intel box
#               (e.g. ./connect_board.sh)
#   DEST        board project root (default /work/bevfusion_migration)
#
# Example:
#   JUMP_PW=secret JUMP_HOST=me@jump.example.com CONNECT_SH=./connect_board.sh \
#     python3 scripts/ssh_nms_transfer.py
import pexpect, sys, re, base64, hashlib, os

JUMP_PW   = os.environ.get('JUMP_PW',     '***REDACTED***')
JUMP_HOST = os.environ.get('JUMP_HOST',   '***REDACTED***')
CONNECT_SH= os.environ.get('CONNECT_SH',  '***REDACTED***')
INTEL_PW  = os.environ.get('INTEL_PW',    '***REDACTED***')
DEST      = os.environ.get('DEST',        '/work/bevfusion_migration')

if '***' in JUMP_PW:
    print("[FATAL] set JUMP_PW / JUMP_HOST / CONNECT_SH env vars (see header).", file=sys.stderr)
    sys.exit(2)

# local staging root = parent of this script's dir (scripts/ -> bevfusion_migration/)
MIG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    'cvcuda_ops/nms/nms.hpp',
    'cvcuda_ops/nms/nms.cpp',
    'cvcuda_ops/nms/test/nms_gold.hpp',
    'cvcuda_ops/nms/test/correctness/test_nms.cpp',
    'cvcuda_ops/nms/test/real/test_nms_real.cpp',
    'cvcuda_ops/nms/test/profile/test_nms_profile.cpp',
    'cvcuda_ops/nms/build_nms.sh',
]

fout = open('/tmp/nms_transfer_dump.txt', 'w', encoding='utf-8')
def log(s):
    fout.write(s); fout.flush()
    sys.stdout.write(s); sys.stdout.flush()

# ---- connect to jump ----
cmd = ('ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
       f'-o ConnectTimeout=15 {JUMP_HOST}')
c = pexpect.spawn(cmd, timeout=90, encoding='utf-8')
i = c.expect(['password:', 'yes/no', pexpect.EOF, pexpect.TIMEOUT])
if i == 1:
    c.sendline('yes'); c.expect('password:')
c.sendline(JUMP_PW)
c.expect([r'\$', r'#', pexpect.TIMEOUT], timeout=30)
log('[local] on jump host\n')

# ---- go to intel host (two-stage: jump pw already sent, connect_board.sh
#      opens an ssh to the Intel board which asks for INTEL_PW) ----
c.sendline(CONNECT_SH)
i = c.expect(['password:', 'yes/no', r'\$', r'#', pexpect.TIMEOUT], timeout=60)
if i == 1:
    c.sendline('yes'); c.expect('password:', timeout=30); i = 0
if i == 0:
    c.sendline(INTEL_PW)
    c.expect([r'\$', r'#', pexpect.TIMEOUT], timeout=60)
c.sendline('echo INTEL_READY'); c.expect('INTEL_READY', timeout=20)
log('[local] on intel host\n')

def run(cmd, marker, timeout=60):
    c.sendline(cmd + f' ; echo {marker}_DONE=$?')
    c.expect(re.escape(marker) + r'_DONE=(\d+)', timeout=timeout)
    return c.match.group(1)

# ---- verify /work writable ----
rc = run(f'touch {DEST}/.xfer_test && rm {DEST}/.xfer_test', 'WRK', timeout=15)
log(f'[local] /work writable check exit={rc}\n')
if rc != '0':
    log('[local] FATAL: /work not writable\n'); fout.close(); sys.exit(1)

# ---- transfer each file ----
results = []
for rel in FILES:
    local = os.path.join(MIG, rel)
    data = open(local, 'rb').read()
    local_md5 = hashlib.md5(data).hexdigest()
    b64 = base64.b64encode(data).decode()
    chunks = [b64[i:i+1500] for i in range(0, len(b64), 1500)]

    tgtdir = os.path.dirname(rel)
    run(f'mkdir -p {DEST}/{tgtdir}', 'MK', timeout=15)
    run('rm -f /tmp/mig.b64', 'RM', timeout=10)
    for idx, ch in enumerate(chunks):
        c.sendline(f'printf %s {ch} >> /tmp/mig.b64; echo CK{idx}')
        c.expect(re.escape(f'CK{idx}') + r'\r?\n', timeout=30)
    c.sendline(
        f'base64 -d /tmp/mig.b64 > {DEST}/{rel} 2>/tmp/dec.err; '
        f'echo DECODE=$?; rm -f /tmp/mig.b64; '
        f'echo MD5V=$(md5sum {DEST}/{rel} | cut -d" " -f1); echo ENDFILE')
    c.expect('DECODE=(\d+)', timeout=30); dec_rc = c.match.group(1)
    c.expect('MD5V=([0-9a-f]{32})', timeout=20); remote_md5 = c.match.group(1)
    c.expect('ENDFILE', timeout=15)
    ok = (dec_rc == '0' and remote_md5 == local_md5)
    results.append((rel, ok))
    log(f'[local] {rel}: {"OK" if ok else "MISMATCH"} '
        f'(local={local_md5} remote={remote_md5} decode={dec_rc})\n')

allok = all(r[1] for r in results)
log('\n[local] === TRANSFER SUMMARY ===\n')
for rel, ok in results:
    log(f'  {"OK " if ok else "BAD"} {rel}\n')
log(f'[local] all files transferred correctly: {allok}\n')

# ---- list the nms tree on the board ----
c.sendline(f'find {DEST}/cvcuda_ops/nms -type f -printf "%p %s\\n" | sort; echo LISTER_DONE')
c.expect('LISTER_DONE', timeout=30)

c.sendline('exit'); c.expect([r'\$', pexpect.TIMEOUT], timeout=15)
c.sendline('exit'); c.expect(pexpect.EOF, timeout=10)
fout.close()
print(f'[local] dump: /tmp/nms_transfer_dump.txt ({os.path.getsize("/tmp/nms_transfer_dump.txt")} B)')
sys.exit(0 if allok else 1)
