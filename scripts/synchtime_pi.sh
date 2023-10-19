offset=0.318
b=`date +%s.%N`; a=`echo $b + $offset|bc`; ssh $pi "echo apollo | sudo -S date --set=\"@$a\""
