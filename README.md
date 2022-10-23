# GDD
GDD: Drifts detection in event logs 

Python packages required
	lxml (http://lxml.de/)
	numpy (http://www.numpy.org/)

How to use **************************************************

Command line usage:

	GDD.py [-w value] [-j value] [-n value] log_file_path
	options:
    		-w window size, integer, default value is 100
    		-j jump distance, integer, default value is 3
    		-n change number, integer, default value is 3
Examples:

 	GDD.py -w 100 -j 3 -n 3 cp5k.mxml
 	GDD.py cp5k.mxml

Note: if you have 'Error reading file ', please using absolute path of log file.

A reference for users **************************************************
If you have any quetions for this code, you can email: leilei_lin@126.com.
We would also be happy to discuss the topic of drifts detection with you.

Thanks **************************************************
Test data is provided by A. Maaradji, M. Dumas, M. La Rosa and A. Ostovar etc.
