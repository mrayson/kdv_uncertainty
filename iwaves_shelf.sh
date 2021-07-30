#!/bin/sh
########################################################################
#
# Shell script to run a suntans test case.
#
########################################################################

#SUNTANSHOME=../../main
SUNTANSHOME=/home/suntans/code/suntans/main
SUN=$SUNTANSHOME/sun
SUNPLOT=$SUNTANSHOME/sunplot
PYTHONEXEC=python

. $SUNTANSHOME/Makefile.in

maindatadir=rundata
datadir=SCENARIOS/test_001
makescript=make_scenario_sunkdv.py
starttime='2017-04-01'
endtime='2017-04-02'
draw_num=1

NUMPROCS=$1

if [ -z "$MPIHOME" ] ; then
    EXEC=$SUN
else
    EXEC="$MPIHOME/bin/mpirun -np $NUMPROCS $SUN"
fi

if [ ! -d $datadir ] ; then
    cp -r $maindatadir $datadir
    echo Creatin input files...
    $PYTHONEXEC scripts/$makescript $datadir $starttime $endtime $draw_num
    echo Creating grid...
    $EXEC -g -vvv --datadir=$datadir
else
    cp $maindatadir/suntans.dat $datadir/.
fi

echo Running suntans...
$EXEC -s -vv --datadir=$datadir 

