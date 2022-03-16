# Copyright (c) AGI.__init__. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# MIT_LICENSE file in the root directory of this source tree.
import getpass

from pexpect import spawn


p = spawn('/opt/cisco/anyconnect/bin/vpn connect vpnconnect.rochester.edu')
p.expect('Username: ')
p.sendline('')
p.expect('Password: ')
p.sendline(getpass.getpass())
p.expect('Second Password: ')
p.sendline('push')
p.expect('VPN>')

# To disconnect:
# p = spawn('/opt/cisco/anyconnect/bin/vpn disconnect')
# p.expect('b')
