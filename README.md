# 사용 방법

1. python3 maze_create.py --seed1 <value> --seed2 <value> --tilt <angle>
  - (example) python3 maze_create_251119.py --seed1 4 --seed2 6 --tilt -45

2. python3 path_planning.py
  - A* 알고리즘 이용하여 path planning 후 waypoint 저장 -> planned_path.txt
     
3. python3 main.py --control_mode position --w_maze
  (주의) controll.cpp의 loadPath() 함수의 경로 재설정
3.1 실행
   1. key(2)번을 통해 init_position 이동
   2. key(4)번을 통해 미로 init position 이동
   3. key(5)번을 통해 미로 풀이
