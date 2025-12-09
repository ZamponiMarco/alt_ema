"""
Generate random queuing networks and save them to disk.

This script creates random closed queuing networks with configurable parameters
and saves them as pickle files for later use.
"""

import os
import argparse
import numpy as np

from libs.qn.examples.closed_queuing_network import random_qn
from libs.qn.examples.controller import constant_controller
from libs.qn.model.queuing_network import ClosedQueuingNetwork
from libs.qn.examples.controller import autoscalers


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate random queuing networks and save them to disk.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gen_random_qn.py -o ./output -n 5 -q 20
  python gen_random_qn.py --output-folder data/qns --num-stations 4
  python gen_random_qn.py -o ./output --min-mu 0.1 --max-mu 1.0
        """,
    )

    # Core parameters
    parser.add_argument(
        "-o",
        "--output-folder",
        type=str,
        default="resources/random_qns/",
        help="Output folder for generated queuing networks (default: resources/random_qns/)",
    )
    parser.add_argument(
        "-n",
        "--num-stations",
        type=int,
        default=3,
        help="Number of stations in the queuing network (default: 3)",
    )
    parser.add_argument(
        "-q",
        "--num-networks",
        type=int,
        default=50,
        help="Number of queuing networks to generate (default: 50)",
    )

    # Optional generation parameters
    parser.add_argument(
        "-s",
        "--skewness",
        type=float,
        default=15,
        help="Skewness parameter for random generation (optional)",
    )
    parser.add_argument(
        "-m",
        "--max-users",
        type=int,
        default=80,
        help="Maximum number of users in the system (optional)",
    )
    parser.add_argument(
        "-k",
        "--k-parameter",
        type=float,
        default=2,
        help="K parameter for random generation (optional)",
    )
    parser.add_argument(
        "--min-mu",
        type=float,
        default=1,
        help="Minimum service rate (optional)",
    )
    parser.add_argument(
        "--max-mu",
        type=float,
        default=10,
        help="Maximum service rate (optional)",
    )

    # Filter parameters
    parser.add_argument(
        "--filter-target-max",
        type=float,
        default=0.5,
        help="Maximum value for QN filter (default: 0.5)",
    )
    
    parser.add_argument(
        "--filter-target-min",
        type=float,
        default=0,
        help="Minimum value for QN filter (default: 0)",
    )

    return parser.parse_args()


def print_configuration(args):
    """Print the current configuration."""
    print("\n" + "=" * 60)
    print("CONFIGURATION".center(60))
    print("=" * 60)
    print(f"  Output folder:           {args.output_folder}")
    print(f"  Number of stations:      {args.num_stations}")
    print(f"  Number of networks:      {args.num_networks}")

    print("\n  Optional Parameters:")
    if args.skewness is not None:
        print(f"    Skewness:             {args.skewness}")
    if args.max_users is not None:
        print(f"    Max users:            {args.max_users}")
    if args.k_parameter is not None:
        print(f"    K parameter:          {args.k_parameter}")
    if args.min_mu is not None:
        print(f"    Min mu:               {args.min_mu}")
    if args.max_mu is not None:
        print(f"    Max mu:               {args.max_mu}")

    print("\n  Filter Parameters:")
    print(f"    Filter target (max):   {args.filter_target_max}")
    print(f"    Filter target (min):   {args.filter_target_min}")
    print("=" * 60 + "\n")


def main():
    """Main function to generate random queuing networks."""
    args = parse_arguments()

    FOLDER = args.output_folder
    NUM_STATIONS = args.num_stations
    NUM_NETWORKS = args.num_networks

    # Create output folder
    os.makedirs(FOLDER, exist_ok=True)

    # Print configuration
    print_configuration(args)

    # Define filter function
    def qn_filter(qn):
        """Filter function for valid queuing networks."""
        return (
            np.dot(qn.visit_vector, 1 / qn.mu[1:]) <= args.filter_target_max
            and np.dot(qn.visit_vector, 1 / qn.mu[1:]) >= args.filter_target_min
            and all(qn.visit_vector != 0)
        )

    i = 0

    print(f"Generating {NUM_NETWORKS} queuing networks...\n")

    while i < NUM_NETWORKS:
        qn: ClosedQueuingNetwork = random_qn(
            NUM_STATIONS,
            skewness=args.skewness,
            max_users=args.max_users,
            k=args.k_parameter,
            min_mu=args.min_mu,
            max_mu=args.max_mu,
        )

        if qn_filter(qn):
            qn.save(f"{FOLDER}qn_{i}.pkl")
            print(f"  ✓ Generated QN {i}")
            i += 1

    print()
    print("=" * 60)
    print(f"✓ Successfully generated {i} queuing networks in '{FOLDER}'")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main() 