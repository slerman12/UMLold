# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import getpass
import sys
from pathlib import Path
from pexpect import spawn

pw = getpass.getpass()

if len(sys.argv) > 1:
    try:
        p = spawn('/opt/cisco/anyconnect/bin/vpn connect vpnconnect.rochester.edu')
        p.expect('Username: ')
        p.sendline('')
        p.expect('Password: ')
        p.sendline(pw)
        p.expect('Second Password: ')
        p.sendline('push')
        p.expect('VPN>')
    except:
        pass

experiment = "*"
agent = "*"
suite = "*"

path = f'./Benchmarking/{experiment}/Plots'

(Path(path) / 'bluehive').mkdir(parents=True, exist_ok=True)

p = spawn('sftp slerman@bluehive.circ.rochester.edu')
p.expect('Password: ')
p.sendline(pw)
p.expect('sftp> ')
p.sendline(f"lcd {path}/bluehive")
p.expect('sftp> ')
p.sendline("cd /scratch/slerman/UnifiedML")
p.expect('sftp> ')
p.sendline(f"get {path}/*{experiment}*{agent}*{suite}*png")
p.expect('sftp> ')
p.sendline(f"get {path}/*{experiment}*{agent}*{suite}*json")
p.expect('sftp> ')
