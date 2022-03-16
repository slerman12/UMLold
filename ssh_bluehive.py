# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
from pexpect import spawn
import subprocess
import sys
import getpass

if len(sys.argv) > 1:
    try:
        p = spawn('/opt/cisco/anyconnect/bin/vpn connect vpnconnect.rochester.edu')
        p.expect('Username: ')
        p.sendline('')
        p.expect('Password: ')
        p.sendline(getpass.getpass())
        p.expect('Second Password: ')
        p.sendline('push')
        p.expect('VPN>')
    except:
        pass

host = "slerman@bluehive.circ.rochester.edu"
commands = []
commands.append(f"""cd /scratch/slerman/UnifiedML
git pull origin master""")

# commands.append(f"""cd /scratch/slerman/
# git clone git@github.com:slerman12/UnifiedML.git""")

commands.append(f"""cd /scratch/slerman/UnifiedML
python sbatch.py --sweep_name dmc --ANY_BIGish""")

# params = """Agent=Agents.DQNAgent task=atari/boxing experiment='intense'"""
# commands.append(f"""cd /scratch/slerman/UnifiedML
# python sbatch.py --params "{params}" --ANY_BIGish""")

# commands.append(f"""squeue -u slerman""")

# commands.append(f"""cd /scratch/slerman/UnifiedML
# source /scratch/slerman/miniconda/bin/activate agi
# python Plot.py 'Exp' 'Random 100 Episodes'""")


for command in commands:
    proc = subprocess.Popen(["ssh", host, command],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    result = proc.stdout.readlines()

    if not result:
        error = proc.stderr.readlines()
        print(sys.stderr, "ERROR: %s" % error)
    else:
        print(result)
