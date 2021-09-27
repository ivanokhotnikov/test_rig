import os

PROJECT_ROOT_DIR = "."
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, 'images')
MODELS_PATH = os.path.join(PROJECT_ROOT_DIR, 'models')
DATA_PATH = os.path.join(PROJECT_ROOT_DIR, 'rig_data')
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(MODELS_PATH, exist_ok=True)

FEATURES = [
    'TIME',
    'STEP',
    'HSU DEMAND',
    'M1 SPEED',
    'M1 CURRENT',
    'M1 TORQUE',
    'PT4 SETPOINT',
    'PT4',
    'D1 RPM',
    'D1 CURRENT',
    'D1 TORQUE',
    'M2 RPM',
    'M2 Amp',
    'M2 Torque',
    'CHARGE PT',
    'CHARGE FLOW',
    'M3 RPM',
    'M3 Amp',
    'M3 Torque',
    'Servo PT',
    'SERVO FLOW',
    'M4 ANGLE',
    'HSU IN',
    'TT2',
    'HSU OUT',
    'M5 RPM',
    'M5 Amp',
    'M5 Torque',
    'M6 RPM',
    'M6 Amp',
    'M6 Torque',
    'M7 RPM',
    'M7 Amp',
    'M7 Torque',
]
FEATURES_NO_TIME = [
    # 'TIME',
    'STEP',
    'HSU DEMAND',
    'M1 SPEED',
    'M1 CURRENT',
    'M1 TORQUE',
    'PT4 SETPOINT',
    'PT4',
    'D1 RPM',
    'D1 CURRENT',
    'D1 TORQUE',
    'M2 RPM',
    'M2 Amp',
    'M2 Torque',
    'CHARGE PT',
    'CHARGE FLOW',
    'M3 RPM',
    'M3 Amp',
    'M3 Torque',
    'Servo PT',
    'SERVO FLOW',
    'M4 ANGLE',
    'HSU IN',
    'TT2',
    'HSU OUT',
    'M5 RPM',
    'M5 Amp',
    'M5 Torque',
    'M6 RPM',
    'M6 Amp',
    'M6 Torque',
    'M7 RPM',
    'M7 Amp',
    'M7 Torque',
]
FEATURES_NO_TIME_AND_COMMANDS = [
    # 'TIME',
    'STEP',
    # 'HSU DEMAND',
    'M1 SPEED',
    'M1 CURRENT',
    'M1 TORQUE',
    # 'PT4 SETPOINT',
    'PT4',
    'D1 RPM',
    'D1 CURRENT',
    'D1 TORQUE',
    'M2 RPM',
    'M2 Amp',
    'M2 Torque',
    'CHARGE PT',
    'CHARGE FLOW',
    'M3 RPM',
    'M3 Amp',
    'M3 Torque',
    'Servo PT',
    'SERVO FLOW',
    'M4 ANGLE',
    'HSU IN',
    'TT2',
    'HSU OUT',
    'M5 RPM',
    'M5 Amp',
    'M5 Torque',
    'M6 RPM',
    'M6 Amp',
    'M6 Torque',
    'M7 RPM',
    'M7 Amp',
    'M7 Torque',
]

FOLDS = 5
SEED = 2
VERBOSITY = 100
EARLY_STOPPING_ROUNDS = 100
OPTIMIZATION_TIME_BUDGET = 5 * 60 * 60