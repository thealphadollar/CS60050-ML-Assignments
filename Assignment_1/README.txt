To run the program, please use the following steps.

- Install Python 3.6 with pip
- Run `pip3 install -r requirements.txt`
- Run `python3 solution.py`

To run solution for a particular part of the problem, please comment out rest of the parts from the end of the file in solution.py

Once the solution is running, all the requested graphs as per the question will come up one by one, please close the graphs to move forward. The program runs in the order
of questions asked in the assignment. Please do read the report for more information.

NOTE: The lasso and ridge parts take quite sometime (around 15 minutes) since the criteria for convergence is pretty tight. For testing them, please relax the 
criteria for convergence on Line Number: 282 and 195 for ridge and lasso respectively.

If the fitting loop takes time (doesn't take much time, around 2-3 minutes), please relax the criteria for convergence on line number 99. Except part 1b all parts do not take much time and can be run independently since all information is stored in json files.
