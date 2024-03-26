import os


results_folder = os.path.abspath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "temp"
    )
)
if not os.path.exists(results_folder):
    os.mkdir(results_folder)
