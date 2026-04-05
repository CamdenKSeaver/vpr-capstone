If using mamba/conda:
* Create environment with `mamba env create -f environment.yml`
* To add new dependencies, update the `environment.yml`, then update the environment with `mamba env update -f environment.yml`

If using pip/venv:
* Install the dependencies with `pip install -r requirements.txt`

Our data is currently small enough to be stored in the repository, but refrain from storing anything too large there.

Set up Tableau Desktop locally via https://www.tableau.com/products/desktop-free/download. From the application, you should be able to open our Dashboard/dashboard.twb file, committing updates to the repository when significant changes are made.
* There's potentially a licensed version available at https://www.tableau.com/academic/students, which may work.