
# Installation guide

This install guide explains the setup required to run Jupyter Notebooks with a Matlab backend.

First all you need a working Matlab installation, but you will also need Python.

## Install Python

Secondly you need a Python installation in which you can install packages. In case you are using a system-wide 
installation of Python installing additional packages may require root access. It is highly recommended to install
a new version of Python in your home directory and use that one instead.

You can do this by typing the following commands:
```
wget https://repo.continuum.io/miniconda/Miniconda3-3.5.2-Linux-x86_64.sh
bash Miniconda3-3.5.2-Linux-x86_64.sh
```
When prompted to add the PATH to your ``.bashrc`` file choose yes. Logout and login or repeat the export of PATH in 
your current shell.

Please verify that `which python` points to your recently installed miniconda Python. The command `python --version` 
should report the following: "Python 3.5.2 :: Continuum Analytics, Inc.". 

## Installing Python packages

Once you have installed Python, type the following commands to install required packages:
```
pip install numpy
pip install jupyter
pip install pymatbridge
pip install matlab_kernel
```

## Install MATLAB Engine API for Python

Because we will be executing Matlab code from a Python program, we need to install the Matlab Engine for Python.

You can do this by going to the right directory, which you can find inside your Matlab installation:
```
cd "matlabroot"/extern/engines/python
```
In case you don't know what your "matlabroot" is, you can use the command ``which matlab`` to find where your matlab is 
installed. Remove the final "bin/matlab" part to get to your matlabroot. 

Inside this directory you should see only a directory called 'dist' and a file called 'setup.py'. Use the following
command to install the Matlab Engine for Python:
```
python setup.py build --build-base="builddir" install
```
Where "builddir" is some directory you have write access to, any temporary directory in your home dir would suffice.

## Tying everything together

Next, we need to tell Python where it can find Matlab by setting an evironment variable, execute the following command:
```
export MATLAB_EXECUTABLE=`which matlab`
```
You may want to save this export command to your ``~/.bashrc`` file.

Finally, we need to install the matlab_kernel for Jupyter, using the command:
```
python -m matlab_kernel install --user
```

## Starting the notebook server

Go to the directory where the *.ipynb files are located on your filesystem and type:
```
jupyter notebook
```

This should start a Firefox browser (assuming you ssh'ed to the machine with X-forwarding enabled). 

If you run into trouble with starting a server because starting a browser won't work, you can instead use the command: 
```
jupyter notebook --no-browser
```

On the page that opened, click on the "GPU_FFTs_in_Matlab.ipynb" file to start the notebook for the first hands-on 
session. Try to execute the first code block to see if everything is working.



## Using SSH-tunneling to connect to notebook server

If you can't run firefox on your remote server, or if this is too slow, it is also possible to run firefox locally 
and create an SSH tunnel to remote server.

Use the following command:
```
ssh -N -f -L localhost:8000:localhost:8888 remote_user@remote_host
```
Replace remote_user and remote_host with your username and the hostname of your GPU server.

In your browser navigate to http://localhost:8000, you **do not** need to configure your browser to use a proxy.

In case you run into trouble, look here: https://coderwall.com/p/ohk6cg/remote-access-to-ipython-notebooks-via-ssh

