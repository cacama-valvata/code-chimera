## Totally stole all of this: 
# http://stackoverflow.com/questions/20472288/python-ssh-password-auth-no-external-libraries-or-public-private-keys
#!/usr/bin/python

import sys, pdb, getpass
from socket import gaierror, error, timeout
import threading
sys.path.append('./paramiko/site-packages/')
import paramiko, ecdsa
#try:
#    import interactive
#except ImportError:
#    from . import interactive

COMMAND='hostname; uptime'
USER='secscan1'
HOST='128.114.2.163'
PASSWORD = getpass.getpass("Enter password for SSH systems:")
up = open("up_servers.txt",'w')
ssh_servers = open('20140617_hosts.txt','r')

def scan_ssh(HOST):
  try:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    #print('*** Connecting to: %s ***' % HOST)
    client.connect(hostname=HOST, port=22, username=USER, password=PASSWORD,timeout=5)
    up.write("HOST: %s UP\n" % HOST)
    #chan = client.invoke_shell()
    #stdout = client.exec_command(COMMAND)
    #chan.close()
    client.close()

  except gaierror as e:
    #print('%s: Caught exception: %s: %s *** ' % (HOST,e.__class__, e))
    if (e[0] == 8):
      up.write("HOST: %s DNSINACTIVE\n" % HOST)
    else:
      up.write("HOST: %s :UNCAUGHT EXCEPT: %s\n" % (HOST,e))
  
  except paramiko.AuthenticationException as e:
    #print('%s: Caught exception: %s: %s *** ' % (HOST,e.__class__, e))
    up.write("HOST: %s AUTHFAIL\n" % HOST)

  except timeout as e:
    #print('%s: Caught exception: %s: %s *** ' % (HOST,e.__class__, e))
    up.write("HOST: %s PORTBLOCKED\n" % HOST)

  except error as e:
    #print('%s: Caught exception: %s: %s *** ' % (HOST,e.__class__, e))
    if (e[0] == 60):
      up.write("HOST: %s PORTBLOCKED\n" % HOST)
    else:
      up.write("HOST: %s :UNCAUGHT EXCEPT: %s\n" % (HOST,e))
  
  except exceptions.NotImplementedError:
      up.write("HOST: %s URANDMISSING\n" % HOST)
    

  except Exception as e:
    print('%s: Caught exception: %s: %s *** ' % (HOST,e.__class__, e))
    up.write("HOST: %s :UNCAUGHT EXCEPT: %s\n" % (HOST,e))
    try:
      client.close()
    except:
      pass
  return


def main():
  thread_list = []
  for line in ssh_servers:
    HOST = line.strip()+".ucsc.edu"
    t = threading.Thread(target=scan_ssh, args=(HOST,))
    t.start()
    thread_list.append(t)
  
  for t in thread_list:
    t.join()

  up.close()
  ssh_servers.close()

main()
