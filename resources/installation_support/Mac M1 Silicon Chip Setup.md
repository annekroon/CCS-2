### Mac M1 Silicon Chip Python Library Installation Issue

If you run into any difficulties installing NLP libraries on your macbook, you may have to use an alternative python installation method via a `virtual environment`.
To do this, we will download a lite, community-maintained version of anaconda called `miniforge`.

Open up you terminal and enter the following command:

`chmod +x ~/Downloads/Miniforge3-MacOSX-arm64.sh`

This should download a file in your downloads folder.

Next, we will execute this file using the following command:

`sh ~/Downloads/Miniforge3-MacOSX-arm64.sh`

A prompt will appear, enter `q` followed by `yes`. Now you will be asked for a folder location where you would like to install `miniforge`. Enter the following command and replace the location with your desired storage directory.

`/Users/<your_username>/miniforgeX`

Enter `yes` into any prompt that comes up then restart the terminal app (completely quit by right-clicking the dock icon).

Reopen the terminal app and create a virtual environment by typing in the following:

***Note**: The text between `-n` and `python` is the name of the virtual environment. You can call it whatever you like. You can also specify the python version you want to use (in our case, 3.9).*

`conda create -n CCS-2 python=3.9`

Enter `yes` into any prompt that comes up.

Now you can activate your virtual environment at any time using the following command:

`conda activate CCS-2`

You can deactivate the environment using the following command:

`conda deactivate CCS-2`

When your virtual environment is activated, you can use both `conda` and `pip` commands to install libraries into your virtual environment as you normally would.

***FOR JUPYTER NOTEBOOK**: Remember to change the python kernel to the one associated with your virtual environment*

For detailed instruction, please use this guide (in incognito mode to avoid paywalls):
<https://towardsdatascience.com/setting-up-apples-new-m1-macbooks-for-machine-learning-f9c6d67d2c0f?gi=10295ce77f5>
