#!/bin/sh
# cargo build --release # re-compile 

rm log.txt # clear the log 
# continuously search for arbitrages
while true
do
    echo "------" >> log.txt
    echo $(date) >> log.txt
    echo "------" >> log.txt
    ./target/release/main --cluster mainnet >> log.txt 2>&1
done