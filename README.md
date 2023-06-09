# About

Tool to allow engineers to easily calculate TCP and WRF for robots which this standard feature is not avaiable. The following tool can be used on the HMI application or stand alone. 

**NOTE:** Meca are not mechanical calibrated robots. The TCP and WRF could differ from the calculated coordinates acquired using this application to the current robot mechanical TCP position.

# Application requirement

The minimal requirement to run the exe file:

- Windows 10

# Opening Application

Open explorer and find a file called: "TCP_WRF_Calculator_v*.*.*" (note that the version may vary) marked as [1].

![Opening the exe file on a windows machine](images/img1.png "Opening the exe file on a windows machine")

# Application Overview
 
Application contains two main sections:

**A section highlighted in red**
Section for TCP Calculations – marked red and with numbers [1], [2], [3]

​​​​​​​**B section highlighted in green**
Section for WRF Calculations – marked green and with numbers [4], [5], [6]

![TCP / WRF Calculator Overview](images/img2.png "TCP / WRF Calculator Overview")

[1] Input fields for all positions used in TCP calculations. **Note:** all the fields need to be populated to perform calculations.

[2] Output text fields – displays the current status of TCP calculations. Text is changed if: no calculations performed yet, the error occurs, or calculations are completed successfully.

[3] Button used to trigger calculations for TCP.

[4] Input fields for all positions used in WRF calculations. **Note:** all the fields need to be populated to perform calculations.

[5] Output text fields – displays the current status of WRF calculations. Text is changed if: no calculations performed yet, the error occurs, or calculations are completed successfully.

[6] Button used to trigger calculations for WRF.

# TCP Calculations

1. Using manual control of a robot get 10 different positions.
**Note:** Make sure the positions are not too close to each other. 

2. Populate values for X, Y, Z, RX, RY, RZ for all 10 positions [1].
**Note:** Make sure all the positions and values are populated, if necessary, insert 0 but do not leave any input field empty.

3. If input field/fields [1] is/are empty an error message will be displayed [2] “No value calculated for … Please enter valid numbers!".

![TCP Calculations - No all data entered](images/img3.png "TCP Calculations - No all data entered")

---

1. Confirm all input fields are populated [1].

2. Select “Calculate TCP” button [2] to begin calculations.

![Trigger TCP calculations](images/img4.png "Trigger TCP calculations")

---

1. Check the result in X, Y, Z format on [1].

2. Come back to the robot configuration software/interface and enter the received values.

![TCP Result](images/img5.png "TCP Result")

# WRF Calculations

1. Using manual control of a robot get 3 different positions.
**Note:** Make sure the positions are not too close to each other. 

2. Populate values for X, Y, Z, RX, RY, RZ for all 3 positions [1].
**Note:** Make sure all the positions and values are populated, if necessary, insert 0 but do not leave any input field empty.
​​​​​​​
3. If input field/fields [1] is/are empty an error message will be displayed [2] “No value calculated for … Please enter valid numbers!".

![WRF Calculations - No all data entered](images/img6.png "WRF Calculations - No all data entered")

---

1. Confirm all input fields are populated [1].
​​​​​​​
2. Select “Calculate WRF” button [2] to begin calculations.

![Trigger WRF calculations](images/img7.png "Trigger WRF calculations")

---

1. Check the result in X, Y, Z format on [1].
​​​​​​​
2. Come back to the robot configuration software/interface and enter the received values.

![WRF Results](images/img8.png "WRF Results")


