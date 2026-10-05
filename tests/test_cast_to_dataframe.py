from pathlib import Path
from rdn.rebuttal.experiment import Result, Parameters, run_experiment

def test_cast_to_dataframe():

    PATH_TO_PAR_FOLDER = Path(__file__).parent / "parameters/"
    PATH_TO_SAVE_FOLDER = Path(__file__).parent / "output/"

    results = []

    for file in PATH_TO_PAR_FOLDER.iterdir():
        path_to_parameters = PATH_TO_PAR_FOLDER / file
        result = run_experiment(path_to_parameters, PATH_TO_SAVE_FOLDER)
        results.append(result)

    Result.cast_spines_to_dataframe(results)

    

