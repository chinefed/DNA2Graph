import os
import sys
import json
import argparse
import multiprocessing as mp

from tqdm import tqdm

from dna2graph.constants import (
    APP_NAME,
    CLI_COMMAND,
    DEFAULT_CONFIG_FILENAME,
    INPUT_EXT
)
from dna2graph.utils import (
    get_asset_path,
    check_for_updates
)
from dna2graph.core.app import DNA2Graph


def build_parser():
    parser = argparse.ArgumentParser(
        prog=CLI_COMMAND,
        description=f'{APP_NAME} command-line interface.'
    )

    # Input directory
    parser.add_argument(
        '-i',
        '--input-dir',
        required=True,
        help='Directory with the images to process.',
    )

    # Output directory
    parser.add_argument(
        '-o',
        '--output-dir',
        required=True,
        help='Root output directory.',
    )

    # Configuration
    default_config_path = get_asset_path(DEFAULT_CONFIG_FILENAME)
    parser.add_argument(
        '--config',
        required=False,
        default=default_config_path,
        help='Path to custom configuration file.',
    )

    # Use trained segmentation pipeline
    parser.add_argument(
        '--trained-seg',
        action='store_true',
        help='Use trained segmentation pipeline.'
    )

    # Save options
    parser.add_argument(
        '-G',
        '--save-graph',
        action='store_true',
        help='Save graph representation.'
    )
    parser.add_argument(
        '-M',
        '--save-mask',
        action='store_true',
        help='Save segmentation mask.'
    )
    parser.add_argument(
        '-R',
        '--save-report',
        action='store_true',
        help='Save report.'
    )
    parser.add_argument(
        '-S',
        '--save-segmentation-rois',
        action='store_true',
        help='Save segmentation ROIs in ImageJ format.'
    )
    parser.add_argument(
        '-B',
        '--save-bbox-rois',
        action='store_true',
        help='Save bounding box ROIs in ImageJ format.'
    )
    parser.add_argument(
        '-L',
        '--save-lin-decomp-rois',
        action='store_true',
        help='Save linear decomposition ROIs in ImageJ format.'
    )

    # Cache
    parser.add_argument(
        '--clean-cache',
        action='store_true',
        help='Clean the cache and process from scratch.'
    )

    return parser


def validate_args(args):
    # Check that the input and output directories exist
    if not os.path.isdir(args.input_dir):
        print(
            'ERROR: Input directory does not exist.',
            file=sys.stderr
        )
        sys.exit(1)

    if not os.path.isdir(args.output_dir):
        print(
            'ERROR: Output directory does not exist.',
            file=sys.stderr
        )
        sys.exit(1)

    # Check that the configuration file exists
    if not os.path.isfile(args.config):
        print(
            'ERROR: Configuration file does not exist.',
            file=sys.stderr
        )
        sys.exit(1)

    # Check that at least one output option is specified
    save_flags = [
        args.save_graph,
        args.save_mask,
        args.save_report,
        args.save_segmentation_rois,
        args.save_bbox_rois,
        args.save_lin_decomp_rois
    ]
    if not any(save_flags):
        print(
            'ERROR: You must specify at least one output option.',
            file=sys.stderr
        )
        sys.exit(1)


def main():
    mp.freeze_support()
    
    # Check for updates
    check_for_updates()

    # Parse and validate the arguments
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)

    # Define stop event that aborts all the processes created
    # by the program in case the user interrupts its execution
    stop_event = mp.Event()

    # Load the configuration file
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
    except:
        print(
            'ERROR: Unable to load the configuration file.',
            file=sys.stderr
        )
        sys.exit(1)
    
    # Initialize the DNA2Graph app
    dna2graph = DNA2Graph(
        output_root_dir=args.output_dir,
        config=config,
        trained_seg=args.trained_seg,
        save_graph=args.save_graph,
        save_mask=args.save_mask,
        save_report=args.save_report,
        save_segmentation_rois=args.save_segmentation_rois,
        save_bbox_rois=args.save_bbox_rois,
        save_lin_decomp_rois=args.save_lin_decomp_rois,
        clean_cache=args.clean_cache,
        stop_event=stop_event
    )

    # Get the paths of the images to process
    img_paths = [
        item.path for item in os.scandir(args.input_dir)
        if item.is_file() and item.name.lower().endswith(INPUT_EXT)
    ]
    
    # Check that in the input directory there is at least one image to process
    n_image_paths = len(img_paths)
    if n_image_paths == 0:
        print(
            'ERROR: Input folder does not contain any image. ' \
            f'Supported extensions: {INPUT_EXT}',
            file=sys.stderr
        )
        sys.exit(1)

    # Process the images
    for img_path in tqdm(img_paths, desc='Processing images', unit='image'):
        try:
            dna2graph.forward(img_path)
        except KeyboardInterrupt:
            stop_event.set()
            tqdm.write('Interrupted by user. Shutting down...', file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            tqdm.write(
                f'ERROR: Unable to process image: {img_path}\n{e}\n'
                'Skipping to the next image.\n',
                file=sys.stderr
            )

    print('Analysis completed!')


if __name__ == '__main__':
    main()