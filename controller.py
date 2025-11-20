import time
import mujoco
import mujoco.viewer
import numpy as np
import os
import sys
import threading
current_dir = os.path.dirname(os.path.abspath(__file__))
if os.name == 'nt':
    import msvcrt
    target_path = os.path.join(current_dir, "build", "Release")
else:
    import select
    import termios
    import tty
    target_path = os.path.join(current_dir, "build")
sys.path.append(target_path)
from cRoboticsController_wrapper_cpp import cRoboticsController as cRoboticsControllerCPP


def precise_sleep(duration):
    start = time.perf_counter()
    while True:
        now = time.perf_counter()
        if (now - start) >= duration:
            break

class RobotController():
    def __init__(self, manipulator_control_mode: str, w_maze: bool):

        self.manipulator_control_mode = manipulator_control_mode

        # Load the model
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(base_dir, "franka_fr3")

        self.w_maze = w_maze

        if self.w_maze:
            self.xml_file_path = os.path.join(model_dir, f"{self.manipulator_control_mode}_w_maze_scene.xml")
            self.urdf_file_path = os.path.join(model_dir, "fr3_w_bar.urdf")
        else:
            self.xml_file_path = os.path.join(model_dir, f"{self.manipulator_control_mode}_scene.xml")
            self.urdf_file_path = os.path.join(model_dir, "fr3.urdf")

        self.mujoco_model = mujoco.MjModel.from_xml_path(self.xml_file_path)

        self.ee_site_id = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")

        # --- Maze sensor tracking (only if maze is present) ---
        if self.w_maze:
            gate_pairs = {
                "maze_1_entrance": ("maze_1_entrance_a_sensor", "maze_1_entrance_b_sensor"),
                "maze_1_exit":     ("maze_1_exit_a_sensor",     "maze_1_exit_b_sensor"),
                "maze_2_entrance": ("maze_2_entrance_a_sensor", "maze_2_entrance_b_sensor"),
                "maze_2_exit":     ("maze_2_exit_a_sensor",     "maze_2_exit_b_sensor"),
            }
            self.gate_names = list(gate_pairs.keys())
            self.gate_ids = {}
            for gate, (name_a, name_b) in gate_pairs.items():
                ida = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, name_a)
                idb = mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, name_b)
                if ida < 0 or idb < 0:
                    print(f"[WARN] gate '{gate}' sensors not found: {name_a}(id={ida}), {name_b}(id={idb})")
                self.gate_ids[(gate, "A")] = ida
                self.gate_ids[(gate, "B")] = idb

            # ----- Interior volume sensors (maze interior under top walls) -----
            self.inside_sensor_ids = {
                "maze_1": mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, "maze_1_inside_sensor"),
                "maze_2": mujoco.mj_name2id(self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, "maze_2_inside_sensor"),
            }

            # ★ Corridor inside sensors
            self.corridor_inside_sensor_ids = {
                "maze_1_exit_corridor": mujoco.mj_name2id(
                    self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, "maze_1_exit_corridor_inside_sensor"
                ),
                "maze_2_entrance_corridor": mujoco.mj_name2id(
                    self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, "maze_2_entrance_corridor_inside_sensor"
                ),
            }

            self.valid_region_geom_ids = []
            for gid in [
                self.inside_sensor_ids["maze_1"],
                self.corridor_inside_sensor_ids["maze_1_exit_corridor"],
                self.corridor_inside_sensor_ids["maze_2_entrance_corridor"],
                self.inside_sensor_ids["maze_2"],
            ]:
                if gid is None or gid < 0:
                    continue
                self.valid_region_geom_ids.append(gid)

            # Last region where the EE was for each gate: None / "A" / "B"
            self.gate_last_region = {gate: None for gate in self.gate_names}
            # Passage times for A→B / B→A
            self.gate_forward_time = {gate: None for gate in self.gate_names}   # A->B
            self.gate_backward_time = {gate: None for gate in self.gate_names}  # B->A

            # Maze entrance/exit times and success flags
            self.maze_entry_time = {"maze_1": None, "maze_2": None}
            self.maze_exit_time  = {"maze_1": None, "maze_2": None}
            self.maze_success    = {"maze_1": False, "maze_2": False}

            self.outside_total_time = 0.0
            self.inside_prev_global = None


        self.dt = self.mujoco_model.opt.timestep

        self.setup_controller()
        
        self.mujoco_data = mujoco.MjData(self.mujoco_model)
        
        # Initial Joint space state
        self.q = np.zeros(len(self.mujoco_mani_joint_names))
        self.qdot = np.zeros(len(self.mujoco_mani_joint_names))
        self.tau = np.zeros(len(self.mujoco_mani_joint_names))
                
        # Manipulator desired positions initialized
        self.q_desired = np.zeros(len(self.mujoco_mani_joint_names))
        self.tau_desired = np.zeros(len(self.mujoco_mani_joint_names))

        self.viewer_fps = 60
        self.print_fps = 2

        self.paused = False
        self.quit = False
        
        # ----- Collision logging state -----
        self.collide_time = 0.0            # Total collision time
        self.in_collision_prev = False     # Whether there was collision in the previous step
        self.first_collision_time = None   # Time of the first collision
        self.last_collision_print_time = 0.0
        self.collision_print_interval = 1.0  # Logging interval during collision [s]

        if os.name != 'nt':
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
        keyThread = threading.Thread(target=self.keyCallback)
        keyThread.daemon = True 
        keyThread.start()
        
    def setup_controller(self):
        self.controller = cRoboticsControllerCPP(self.urdf_file_path, self.manipulator_control_mode, self.dt)
        joint_names = [self.mujoco_model.names[self.mujoco_model.name_jntadr[i]:].split(b'\x00', 1)[0].decode('utf-8') for i in range(self.mujoco_model.njnt) if self.mujoco_model.names[self.mujoco_model.name_jntadr[i]:].split(b'\x00', 1)[0].decode('utf-8')]
        self.q_names = joint_names
        self.v_names = joint_names
        self.actuator_names = [self.mujoco_model.names[self.mujoco_model.name_actuatoradr[i]:].split(b'\x00', 1)[0].decode('utf-8') for i in range(self.mujoco_model.nu)]
        self.mujoco_mani_joint_names = [name for name in self.actuator_names if 'joint' in name]
        
    def updateJointState(self):
        # Get manipulator angle positions, velocities and torques from mujoco
        for i, mani_joint_name in enumerate(self.mujoco_mani_joint_names):
            self.q[i] = self.mujoco_data.qpos[self.q_names.index(mani_joint_name)]
            self.qdot[i] = self.mujoco_data.qvel[self.v_names.index(mani_joint_name)]
            self.tau[i] = self.mujoco_data.qfrc_actuator[self.v_names.index(mani_joint_name)] + \
                        self.mujoco_data.qfrc_applied[self.v_names.index(mani_joint_name)] + \
                        self.mujoco_data.qfrc_constraint[self.v_names.index(mani_joint_name)]
            
    def updateModel(self, q:np.array, qdot:np.array, tau:np.array):
        self.controller.updateModel(q, qdot, tau)
    
    def keyCallback(self):
        while True:
            if os.name == 'nt':
                if msvcrt.kbhit():
                    keycode = msvcrt.getch().decode('utf-8')
                    self.handle_key_input(keycode)
            else:
                if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
                    keycode = sys.stdin.read(1)
                    self.handle_key_input(keycode)

    def handle_key_input(self, keycode):
        if keycode == ' ':
            self.paused = not self.paused
        elif keycode == 'q':
            self.quit = True
        else:
            self.controller.keyMapping(ord(keycode))
            
    def run(self):
        with mujoco.viewer.launch_passive(self.mujoco_model, self.mujoco_data, show_left_ui=False, show_right_ui=False) as viewer:
            last_viewr_update_time = 0
            last_print_update_time = time.perf_counter()
            
            if self.w_maze:
                # Reset collision statistics at the beginning of run
                self.collide_time = 0.0
                self.in_collision_prev = False
                self.first_collision_time = None
                self.last_collision_print_time = 0.0

            # Reset outside-time statistics at the beginning of run
                self.outside_total_time = 0.0
                self.inside_prev_global = None


            while viewer.is_running() and not self.quit:
                if not self.paused:

                    #  ------------------------
                    step_start = time.perf_counter()
                    mujoco.mj_step(self.mujoco_model, self.mujoco_data)

                    # ----- Sensor volumes: gate passage (entrance/exit) and interior region -----
                    if self.w_maze:
                        self.check_sensor_planes()
                        # ----- Collision logging -----
                        if self.mujoco_data.time > 0.5:
                            ncon = self.mujoco_data.ncon
                            if ncon > 0:
                                # Integrate collision time
                                self.collide_time += self.dt

                                now_t = self.mujoco_data.time
                                if not self.in_collision_prev:
                                    # Collision started
                                    self.first_collision_time = (
                                        self.first_collision_time
                                        if self.first_collision_time is not None else now_t
                                    )
                                # Print detailed collision info only at intervals
                                if now_t - self.last_collision_print_time > self.collision_print_interval:
                                    contact = self.mujoco_data.contact[0]
                                    g1 = mujoco.mj_id2name(
                                        self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom1
                                    )
                                    g2 = mujoco.mj_id2name(
                                        self.mujoco_model, mujoco.mjtObj.mjOBJ_GEOM, contact.geom2
                                    )
                                    print(
                                        f"\033[1;31m[COLLISION] t = {now_t:.3f} s, "
                                        f"#contacts = {ncon}, "
                                        f"example: {g1} <-> {g2}\033[0m"
                                    )
                                    self.last_collision_print_time = now_t

                                self.in_collision_prev = True
                            else:
                                if self.in_collision_prev:
                                    # Print once at collision end
                                    now_t = self.mujoco_data.time
                                self.in_collision_prev = False

                    # ----- Controller update -----
                    self.updateJointState()
                    self.updateModel(self.q, self.qdot, self.tau)
                    self.play_time = self.mujoco_data.time

                    if time.perf_counter() - last_print_update_time > 1/self.print_fps:
                        # self.controller.printState()
                        last_print_update_time = time.perf_counter()

                    self.controller.compute(self.play_time)
                    if self.manipulator_control_mode == "position":
                        self.q_desired = self.controller.getCtrlInput()
                    elif  self.manipulator_control_mode == "torque":
                        self.tau_desired = self.controller.getCtrlInput()

                    for i, mani_joint_name in enumerate(self.mujoco_mani_joint_names):
                        self.mujoco_data.ctrl[self.actuator_names.index(mani_joint_name)] = (
                            self.q_desired[i] if self.manipulator_control_mode == "position"
                            else self.tau_desired[i]
                        )

                    if self.mujoco_data.time - last_viewr_update_time > 1/self.viewer_fps:
                        viewer.sync()
                        last_viewr_update_time = self.mujoco_data.time

                    time_until_next_step = self.dt - (time.perf_counter() - step_start)
                    if time_until_next_step > 0:
                        precise_sleep(time_until_next_step)

            # =================== Final summary at the end of simulation ===================
            if self.w_maze:
                print("\033[1;36m============== Maze summary ==============")                
                for maze in ["maze_1", "maze_2"]:
                    ent = self.maze_entry_time[maze]
                    ext = self.maze_exit_time[maze]
                    succ = self.maze_success[maze]
                    if ent is not None:
                        print(f"{maze} entered at t = {ent:.3f} s")
                    else:
                        print(f"{maze} not entered")
                    if ext is not None:
                        print(f"{maze} exited  at t = {ext:.3f} s")
                        if ent is not None:
                            print(f"{maze} time (exit - enter) = {ext - ent:.3f} s")
                    else:
                        print(f"{maze} not exited")
                    print(f"{maze} SUCCESS = {succ}")
                    print("")

                print(f"Time outside maze regions = {self.outside_total_time:.3f} s")
                print(f"Collision time = {self.collide_time:.3f} s")
                print("")

                if self.maze_success["maze_1"] and self.maze_success["maze_2"]:
                    total_time = self.maze_exit_time["maze_2"] - self.maze_entry_time["maze_1"]
                    print(f"All mazes SUCCESS total time = {total_time:.3f} s")
                else:
                    print("All mazes not fully completed")

                print("==================================================\033[0m")

    # --------------------------- Sensor helpers ---------------------------

    def _get_ee_pos(self) -> np.ndarray:
        return self.mujoco_data.site_xpos[self.ee_site_id].copy()

    def _point_inside_geom(self, geom_id: int, p: np.ndarray, margin: float = 1e-6) -> bool:
        if geom_id is None or geom_id < 0:
            return False
        center = self.mujoco_data.geom_xpos[geom_id]
        xmat = self.mujoco_data.geom_xmat[geom_id]
        R = xmat.reshape(3, 3)
        local = R.T @ (p - center)
        half = self.mujoco_model.geom_size[geom_id]
        return np.all(np.abs(local) <= half + margin)

    def check_sensor_planes(self):
        if not self.w_maze:
            return

        t = self.mujoco_data.time
        p = self._get_ee_pos()

        # ----- Gate A/B passage at entrance/exit (maze success/timing) -----
        for gate in self.gate_names:
            ida = self.gate_ids.get((gate, "A"), -1)
            idb = self.gate_ids.get((gate, "B"), -1)

            if ida < 0 or idb < 0:
                continue

            inside_A = self._point_inside_geom(ida, p)
            inside_B = self._point_inside_geom(idb, p)

            prev_region = self.gate_last_region[gate]

            if inside_A and not inside_B:
                current_region = "A"
            elif inside_B and not inside_A:
                current_region = "B"
            elif inside_A and inside_B:
                current_region = prev_region if prev_region in ("A", "B") else "A"
            else:
                current_region = None

            # A -> B : forward passage
            if prev_region == "A" and current_region == "B":
                if self.gate_forward_time[gate] is None:
                    self.gate_forward_time[gate] = t

                if gate == "maze_1_entrance" and self.maze_entry_time["maze_1"] is None:
                    self.maze_entry_time["maze_1"] = t
                    print(f"\033[1;32m[Maze 1] ENTERED at t = {t:.3f} s\033[0m")
                elif gate == "maze_2_entrance" and self.maze_entry_time["maze_2"] is None:
                    self.maze_entry_time["maze_2"] = t
                    print(f"\033[1;32m[Maze 2] ENTERED at t = {t:.3f} s\033[0m")

            # B -> A : backward passage
            elif prev_region == "B" and current_region == "A":
                if self.gate_backward_time[gate] is None:
                    self.gate_backward_time[gate] = t

                if gate == "maze_1_exit" and self.maze_exit_time["maze_1"] is None:
                    self.maze_exit_time["maze_1"] = t
                    self.maze_success["maze_1"] = True
                    print(f"\033[1;32m[Maze 1] EXITED at t = {t:.3f} s \033[0m")
                elif gate == "maze_2_exit" and self.maze_exit_time["maze_2"] is None:
                    self.maze_exit_time["maze_2"] = t
                    self.maze_success["maze_2"] = True
                    print(f"\033[1;32m[Maze 2] EXITED at t = {t:.3f} s \033[0m")

            self.gate_last_region[gate] = current_region
            
        # ----- Checking whther EE inside the maze -----
        if self.maze_entry_time["maze_1"] is not None and self.valid_region_geom_ids:
            inside_any = any(self._point_inside_geom(gid, p) for gid in self.valid_region_geom_ids)

            prev_inside = self.inside_prev_global
            if prev_inside is None:
                self.inside_prev_global = inside_any
            else:
                if prev_inside and not inside_any:
                    print("\033[1;33mEE deviated from maze+corridor region! \033[0m")
                elif (not prev_inside) and inside_any:
                    print("\033[1;33mEE came back to maze+corridor region! \033[0m")

            if not inside_any:
                self.outside_total_time += self.dt

            self.inside_prev_global = inside_any

