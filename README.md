# autoroute-manager

## Enviroment setup
### GDAL for Python
We use conda for all of our environment management (mamba is also used and is faster). Make sure that conda is installed. I would strongly recommend creating a file called .condarc and placing it in the same directory as your main conda folder if it does not already exist. It helps avoid common conflicts between the required packages. Your .condarc file should look like this:
```
channel_priority: strict
channels:
  - conda-forge
  - defaults
```
Create a new enviroment from the environment yaml file included like so:

COMING SOON....

```

### GDAL for AutoRoute (Windows)
AutoRoute, as it currently stands, was compiled on Windows with GDAL 2.2.3. Assuming your windows is 64 bit, download gdal-202-1700-x64-core.msi [here](https://www.gisinternals.com/query2.html?content=filelist&file=release-1700-x64-gdal-2-2-3-mapserver-7-0-7.zip). Run the installer. By default, it will try to install GDAL either in Program Files or in the top level C:\.

In a conda prompt, run the following (the webui expects the environment name to be "autoroute"):
```
conda create -n autoroute
conda activate autoroute
```

Now, move the GDAL folder that was created when you ran the msi from where it is to the environment folder. The environment folder typically looks something like “C:\Users\USERNAME\miniconda3\envs\autoroute", where USERNAME is your username. If you used something other than miniconda, the folder may be “conda”, “anaconda”, or just “minconda”. On windows, it may also be located under AppData\mambaforge if using mamba.

Run the following command:
```
cd %CONDA_PREFIX%
mkdir .\etc\conda\activate.d
mkdir .\etc\conda\deactivate.d
type NUL > .\etc\conda\activate.d\env_vars.bat
type NUL > .\etc\conda\deactivate.d\env_vars.bat
```

The above made two directories in your conda environment and two files in each directory. Go to these, edit using something like notepad. Edit the env_vars.bat in activate.d to be something like:

```
set PATH=%CONDA_PREFIX%\GDAL;%PATH%
set GDAL_DATA=%CONDA_PREFIX%\GDAL\gdal-data
```

And the env_vars.bat in deactivate.d to be: 

```
set GDAL_DATA=
set PATH=%PATH:%CONDA_PREFIX%\GDAL;=%
```
Make sure all these paths are correct casing. Don’t delete the semicolons!

Reopen your anaconda prompt. After activating the environment, you will be able to run AutoRoute



### GDAL for AutoRoute (Linux)
AutoRoute, as it currently stands, was compiled on a Linux with GDAL 3.5.1 
In a conda prompt, run the following (the webui expects the environment name to be "autoroute"):
```
conda create -n autoroute
conda activate autoroute
conda install gdal=3.5.1

cd "$CONDA_PREFIX"
mkdir -p etc/conda/activate.d
mkdir -p etc/conda/deactivate.d
touch etc/conda/activate.d/env_vars.sh
touch etc/conda/deactivate.d/env_vars.sh
```

The above made two directories in your conda environment and two files in each directory. Go to these and edit using your favorite text editor. Edit the env_vars.sh in activate.d to be something like:

```
#!/bin/sh
export PATH=$CONDA_PREFIX/include:$PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
```

And the env_vars.sh in deactivate.d to be: 

```
#!/bin/sh
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$CONDA_PREFIX/lib
export PATH=\$PATH:$CONDA_PREFIX/include
unset PKG_CONFIG_PATH
```
Make sure all these paths are correct casing.

Reopen your anaconda prompt. After activating the environment, you will be able to run AutoRoute
