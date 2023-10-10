offset=0.318
b=`date +%s.%N`; a=`echo $b + $offset|bc`; ssh $panda "echo apollo | sudo -S date --set=\"@$a\""
