#!/usr/bin/env python3
# Build + run the SYCL NMS port on the Intel board via the pexpect SSH tunnel,
# mirroring the colleague's scripts/ssh_hostbuild.py / ssh_benchbuild.py.
#
# On the board: cd /work/bevfusion_migration, run cvcuda_ops/nms/build_nms.sh,
# then run the 3 NMS binaries with ONEAPI_DEVICE_SELECTOR=opencl:gpu (and
# VOX_PROFILE=1 for the profile binary). Captures all output to
# /tmp/nms_buildrun_dump.txt.
#
# Credentials: same env vars as ssh_nms_transfer.py (JUMP_PW / JUMP_HOST /
# CONNECT_SH / DEST).
import pexpect, sys, re, base64, os

JUMP_PW   = os.environ.get('JUMP_PW',     '***REDACTED***')
JUMP_HOST = os.environ.get('JUMP_HOST',   '***REDACTED***')
CONNECT_SH= os.environ.get('CONNECT_SH',  '***REDACTED***')
INTEL_PW  = os.environ.get('INTEL_PW',    '***REDACTED***')
DEST      = os.environ.get('DEST',        '/work/bevfusion_migration')

if '***' in JUMP_PW:
    print("[FATAL] set JUMP_PW / JUMP_HOST / CONNECT_SH env vars (see header).", file=sys.stderr)
    sys.exit(2)

N = os.environ.get('NMS_N', '5000')   # num_bboxes for real/profile
S = os.environ.get('NMS_S', '4')      # num_samples

fout = open('/tmp/nms_buildrun_dump.txt', 'w', encoding='utf-8')
def log(s):
    fout.write(s); fout.flush()
    sys.stdout.write(s); sys.stdout.flush()

cmd = ('ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null '
       f'-o ConnectTimeout=15 {JUMP_HOST}')
c = pexpect.spawn(cmd, timeout=90, encoding='utf-8')
i = c.expect(['password:', 'yes/no', pexpect.EOF, pexpect.TIMEOUT])
if i == 1:
    c.sendline('yes'); c.expect('password:')
c.sendline(JUMP_PW)
c.expect([r'\$', r'#', pexpect.TIMEOUT], timeout=30)
c.sendline(CONNECT_SH)
i = c.expect(['password:', 'yes/no', r'\$', r'#', pexpect.TIMEOUT], timeout=60)
if i == 1:
    c.sendline('yes'); c.expect('password:', timeout=30); i = 0
if i == 0:
    c.sendline(INTEL_PW)
    c.expect([r'\$', r'#', pexpect.TIMEOUT], timeout=60)
c.sendline('echo INTEL_READY'); c.expect('INTEL_READY', timeout=20)
log('[local] on intel host\n')

inner = f'''
set +e
cd {DEST}
echo "===== icpx --version ====="
icpx --version 2>&1 | head -3
echo "===== sycl-ls (opencl:gpu) ====="
ONEAPI_DEVICE_SELECTOR=opencl:gpu sycl-ls 2>&1 | head -10
echo "===== BUILD NMS ====="
bash cvcuda_ops/nms/build_nms.sh 2>&1 | tail -20
echo "BUILD_RC=${{PIPESTATUS[0]}}"
echo "===== RUN correctness ====="
ONEAPI_DEVICE_SELECTOR=opencl:gpu ./cvcuda_ops/nms/test/correctness/test_nms 2>&1
echo "CORR_RC=$?"
echo "===== RUN real (N={N} S={S}) ====="
ONEAPI_DEVICE_SELECTOR=opencl:gpu ./cvcuda_ops/nms/test/real/test_nms_real {N} {S} 10 2>&1
echo "REAL_RC=$?"
echo "===== RUN profile (N={N} S={S}) ====="
ONEAPI_DEVICE_SELECTOR=opencl:gpu VOX_PROFILE=1 ./cvcuda_ops/nms/test/profile/test_nms_profile {N} {S} 20 2>&1
echo "PROF_RC=$?"
echo "===== END ====="
'''
b64 = base64.b64encode(inner.encode()).decode()
c.logfile_read = fout
c.sendline(f"echo {b64} | base64 -d | bash ; echo ZNMS_EXIT=$?")
c.expect(re.escape('ZNMS_EXIT=') + r'(\d+)', timeout=900)
c.logfile_read = None

try:
    c.sendline('exit'); c.expect([r'\$', pexpect.TIMEOUT], timeout=15)
    c.sendline('exit'); c.expect(pexpect.EOF, timeout=10)
except Exception:
    pass
fout.close()
print(f'[local] dump: /tmp/nms_buildrun_dump.txt ({os.path.getsize("/tmp/nms_buildrun_dump.txt")} B)')
