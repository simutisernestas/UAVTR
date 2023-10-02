offset=0.318
b=`date +%s.%N`; a=`echo $b + $offset|bc`; ssh $panda "echo panda | sudo -S date --set=\"@$a\""