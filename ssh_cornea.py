# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import subprocess
import sys

HOST = "cornea"

COMMAND = r"""
cd UnifiedML
conda activate agi
python Plot.py
"""

proc = subprocess.Popen(["ssh", HOST, COMMAND],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE)

result = proc.stdout.readlines()

if not result:
    error = proc.stderr.readlines()
    print(sys.stderr, "ERROR: %s" % error)
else:
    print(result)
