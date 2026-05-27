APP_NAME = 'DNA2Graph'
PACKAGE_NAME = 'dna2graph'
CLI_COMMAND = 'dna2graph-cli'

DEVELOPER = 'Federico Chinello'
DEVELOPER_EMAIL = 'federico.chinello@studbocconi.it'

USER_CONFIG_FILENAME = 'config.json'
DEFAULT_CONFIG_FILENAME = 'defaults.json'

TK_STYLE = 'clam'

INPUT_EXT = ('.tif', '.tiff', '.png', '.jpeg', '.jpg')

TRAINED_SUFFIX = '_trained'
SECTIONS_BASE = (
    'validity_mask',
    'segmenter',
    'corrector'
)
SECTIONS_TRAINED = (
    'validity_mask', 
    f'segmenter{TRAINED_SUFFIX}',
    f'corrector{TRAINED_SUFFIX}'
)