# applied-econ-final_project

1. If you prefer running python script, all you need to download are 3 files: requirements.txt, main.py, and function_library.py, and nothing else. Use ```pip install -r requirements.txt``` to recreate the environment. Then, run main.py. 

2. If you prefer running R, you need to download all the files. Then open r-script.R and restore the r environment with 
```renv::restore()```  and run r-script.R. 


## Project structure
If you are running python (which I would recommend): 

1. main.py is the file you run. It runs the simulations and creates all the output pdf graphs in folder ```output``` (eg. "utility-diff-(-0.5)-all both.pdf"...)

2. function.py is the file containing all my functions. main.py imports function.py.

3. requirements.txt provides the versions of python environment and packages you need to run main.py. 

4. unit-test.py tests the first two functions in the function library. 

If you are running R: 

1. r-script.R is the file you run and renv folder, renv.lock are the files tracking the R environment. r-script.R imports a package ```reticulate``` and allow us to call and run .py files. 

2. To successfully run r-script.R, you need python environment as well since this .R file calls .py files. 

