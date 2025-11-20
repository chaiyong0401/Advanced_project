# Detect Conda environment path dynamically
if(WIN32)
    execute_process(
        COMMAND conda env list
        OUTPUT_VARIABLE CONDA_ENV_LIST
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Debugging: Print the raw output of the command
    message(STATUS "Raw output of 'conda env list': ${CONDA_ENV_LIST}")

    # Adjust regex to capture the correct path
    string(REGEX MATCH "C:\\\\Users\\\\[^\\s]+\\\\anaconda3\\\\envs\\\\cRobotics" CONDA_ENV_PATH "${CONDA_ENV_LIST}")
else()
    execute_process(
        COMMAND conda env list
        OUTPUT_VARIABLE CONDA_ENV_LIST
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    # Debugging: Print the raw output of the command
    message(STATUS "Raw output of 'conda env list': ${CONDA_ENV_LIST}")

    # Adjust regex for Linux/Unix paths
    # string(REGEX MATCH "/[^\\s]+/anaconda3/envs/cRobotics" CONDA_ENV_PATH "${CONDA_ENV_LIST}")
    string(REGEX REPLACE "\n" ";" CONDA_ENV_LIST "${CONDA_ENV_LIST}")
foreach(line ${CONDA_ENV_LIST})
    if(line MATCHES ".*cRobotics.*")
        string(REGEX REPLACE " +" ";" tokens "${line}")
        list(GET tokens -1 maybe_path)
        if(EXISTS "${maybe_path}")
            set(CONDA_ENV_PATH "${maybe_path}")
            break()
        endif()
    endif()
endforeach()
endif()

# Ensure a valid path was found
if(CONDA_ENV_PATH)
    set(CONDA_ENV_PATH "${CONDA_ENV_PATH}" CACHE PATH "Path to the cRobotics Conda environment")
    message(STATUS "Detected Conda environment path: ${CONDA_ENV_PATH}")
else()
    # Print raw output for debugging purposes
    message(STATUS "Raw output of 'conda env list': ${CONDA_ENV_LIST}")
    message(FATAL_ERROR "cRobotics environment not found. Please ensure the environment is created.")
endif()
