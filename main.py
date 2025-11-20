import argparse
from controller import RobotController


def main(args):
    VALID_CONTROL_MODES = ["position", "torque"]
    print(f"Control mode: {args.control_mode}")
    if args.control_mode not in VALID_CONTROL_MODES:
        raise ValueError(f"Invalid control_mode '{args.control_mode}'. "
                         f"Must be one of {VALID_CONTROL_MODES}.")
    if args.w_maze:
        print("Maze environment activated!")
    robot_controller = RobotController(args.control_mode, args.w_maze)
    robot_controller.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--control_mode", type=str, default="position", help="control mode [position, torque]")
    parser.add_argument('--w_maze', action='store_true', help='Activate maze environment')

    args = parser.parse_args()
    main(args)