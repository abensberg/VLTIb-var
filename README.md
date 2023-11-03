# VLTI B-VAR
## An interactive tool to calculate and display visibility and closure phase for VLTI observations for different hour angles

## Description
Variability is a vital feature in protoplanetary disks. However, the measured visbility and closure phase of an object can also vary when comparing observations done at different hour angles due to different projected baselines. VLTI B-VAR allows to estimate this effect using four different disk models. It is also possible to upload custom intensity maps from which closure phase and visibilities should be calculated. <br>
A detailed study of the influence of protoplanetary disk variability on the visiblities and closure phases can be found in [Bensberg,Kobus & Wolf 2023](https://ui.adsabs.harvard.edu/abs/2023A%26A...677A.126B/abstract).

## Installation
VLTI B-VAR is ment to be run as Streamlit-App in a browser. In case you want to use it on your local machine, clone the github-repository and run <br>
`pip install -r requirements.txt` <br>
to install the necessary librarys. <br>
To start the app enter <br>
`streamlit run vlti_b-var.py` <br>
into your terminal.

## Usage
Use the slider and buttons to choose your set of parameters for the calculations. In case you want to upload your own custom model intensity map, please make sure that the uploaded file is a numpy file (.npy) containing a 2D numpy array where the dimensions of the image are equal for x- and y-axis.

## Support
Please refer to github or 
<abensberg@astrophysik.uni-kiel.de>.

## Authors and acknowledgment
If you use results from VLTI B-VAR, please cite [Bensberg,Kobus & Wolf 2023](https://ui.adsabs.harvard.edu/abs/2023A%26A...677A.126B/abstract) as well as the website.
