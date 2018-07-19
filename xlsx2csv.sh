#!/bin/bash
# This script converts .xls and xlsx files to csv files using the ssconvert tool provided by gnumeric
# To install gnumeric on ubuntu run
# $ sudo apt install gnumeric
for f in *.xlsx; do 
    ssconvert "$f" "${f%.xlsx}.csv";
done
