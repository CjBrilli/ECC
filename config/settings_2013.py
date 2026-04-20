from pathlib import Path

YEAR = "2013"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent

DOPPLER_FILE = DATA_ROOT / "dataByYear" / f"data_{YEAR}.txt"
HORIZONS_FILE = PROJECT_ROOT / "inputs" / f"{YEAR}_horizons.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / YEAR

F_CARRIER = 8.4e9
C = 299792458.0

T_INT = 60.0
C_BAND = 1.9e-6

MAX_ABS_DOPPLER_HZ = 0.3
MIN_ELEV_DEG = 15.0
MIN_SAMPLES_PER_DAY = 10

RESAMPLE_RULE = "60s"
SMOOTH_DAYS = 7

DT_TARGET = 10.0
F_LOW = 3e-4
F_HIGH = 3e-2
WINDOW_MIN = 20
STEP_MIN = 10
MIN_SAMPLES = 50

USE_VALID_FLAG = True
USE_TROPO_DIAGNOSTIC = True

NOTES = """
2011 baseline year.
"""