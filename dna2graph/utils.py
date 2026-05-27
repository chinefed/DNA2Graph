import sys
import json
from pathlib import Path
from urllib.request import urlopen
from importlib.resources import files
from importlib.metadata import version
from packaging.version import parse

from dna2graph.constants import APP_NAME, PACKAGE_NAME


class UserCancelledError(Exception):
    '''Raised when the user cancels an analysis job.'''
    pass


def get_asset_path(file_name: str) -> Path:
    '''
    Return path to an asset file.
    '''
    if hasattr(sys, '_MEIPASS'):
        # Running inside PyInstaller bundle
        base_path = Path(sys._MEIPASS)
        return base_path / 'assets' / file_name
    else:
        return files('dna2graph.assets') / file_name
    

def print_update_message(current, latest):
    '''
    Print message suggesting user to update the application.
    '''
    banner = "#" * 68
    print(
        f"\n{banner}\n"
        f"### A NEW VERSION OF {APP_NAME.upper()} IS AVAILABLE! ###\n"
        f"{banner}\n"
        "\n"
        f"Current version installed : {current}\n"
        f"Latest version available : {latest}\n"
        "\n"
        f"To update {APP_NAME}, run the following command:\n"
        "\n"
        f"    pip install --upgrade {PACKAGE_NAME}\n"
        "\n"
        "After the update finishes, restart the application.\n"
        f"\n{banner}\n"
    )
    
    
def check_for_updates():
    '''
    Checks for updates on PyPI.
    '''
    try:
        current = version(PACKAGE_NAME)
        with urlopen(
            f'https://pypi.org/pypi/{PACKAGE_NAME}/json',
            timeout=5
        ) as response:
            latest = json.load(response)['info']['version']

        if parse(latest) > parse(current):
                print_update_message(current, latest)

    except Exception:
        pass