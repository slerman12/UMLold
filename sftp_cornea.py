# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from pathlib import Path
from pexpect import spawn

experiment = "*"
agent = "*"
suite = "*"

path = f'./Benchmarking/{experiment}/Plots'
(Path(path) / 'cornea').mkdir(parents=True, exist_ok=True)

p = spawn('sftp cornea')
p.expect('sftp> ')
p.sendline(f"lcd {path}/cornea")
p.expect('sftp> ')
p.sendline("cd UnifiedML")
p.expect('sftp> ')
p.sendline(f"get {path}/*{experiment}*{agent}*{suite}*png")
p.expect('sftp> ')
p.sendline(f"get {path}/*{experiment}*{agent}*{suite}*json")
p.expect('sftp> ')

# path = f'./Benchmarking/{experiment}/Video_Image'
# (Path(path) / 'cornea').mkdir(parents=True, exist_ok=True)
#
# p = spawn('sftp cornea')
# p.expect('sftp> ')
# p.sendline(f"lcd {path}/cornea")
# p.expect('sftp> ')
# p.sendline("cd UnifiedML")
# p.expect('sftp> ')
# p.sendline(f"get ./Benchmarking/{experiment}/{agent}/{suite}/*Video_Image/*png")
# p.expect('sftp> ', timeout=1000000)
