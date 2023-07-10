#intel AI squat count
* URL : https://github.com/openvinotoolkit/openvino_notebooks/tree/main/notebooks
* 205-vision-background-removal
* 402-pose-estimation-webcam
* AI를 활용해 사람을 인지하여 사람을 제외한 뒷 배경을 지우고 사람이 스쿼트를 하면 개수를 세는 프로그램이다.

# Requirement (필요 환경)
* 프로세서	Intel(R) Core(TM) i7-9700 CPU @ 3.00GHz   3.00 GHz
* At least : RAM	32.0GB
*	64bit OS, x64 Processor

# Clone code
* https://github.com/excelsior19999/intel_AI_squat_count/new/main

# Prerequite (전제 조건)
* Install Python 3.9.17 ver
* Install anaconda3
* Create Virtual Environment

* 
* python -m venv .venv
* source .venv/Script/activate

* python -m pip install -U pip
* python -m pip install wheel

* python -m pip install openvino-dev

* open model zoo Download Link:
*  https://github.com/openvinotoolkit/open_model_zoo

* cd C:\Users\AIoT24\open_model_zoo\demos  || or your Path
* python -m pip install -r requirements.txt

# Steps to build
* cd AIoT24\Intel Class\openvino_notebooks\notebooks\402-pose-estimation-webcam

* make
* make install

* Full Paht (just because)
* c:\Users\AIoT24\Intel Class\openvino_notebooks\notebooks\402-pose-estimation-webcam

# Step to run
* Use VSCode, use python
* Ctrl+Shift+p in VSCode
* Python:Select Interpreter
* Python 3.9.17('aiot-24') ~\anaconda3\envs\aiot-24\python.exe



* 참조 : https://github.com/mokiya/kcci.intel.ai.project/tree/main/Round-01/Template
