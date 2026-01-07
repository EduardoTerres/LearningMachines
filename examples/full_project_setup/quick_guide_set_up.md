This is just for refence since you dont have the robot and dont know the ip address etc.
# macOS (Apple Silicon) 

## 1. Simulation

### Terminal 1: Start CoppeliaSim
```bash
cd /Users/liciatauriello/Desktop/Learning_machine/project/repo_robobo/learning_machines_robobo/examples/full_project_setup
zsh ./scripts/start_coppelia_sim.zsh ./scenes/Robobo_Scene.ttt
```
**Keep this terminal open.** CoppeliaSim will launch.

### Terminal 2: Run Your Code
```bash
cd /Users/liciatauriello/Desktop/Learning_machine/project/repo_robobo/learning_machines_robobo/examples/full_project_setup
zsh ./scripts/run_apple_sillicon.zsh --simulation
```
Your code will run in the Docker container.
to check use : docker ps

---


## Quick Start - Hardware (Physical Robot)

### Prerequisites (Before Running)
1. Connect robot to phone via Bluetooth (easy already paired)
2. Open Robobo app on phone (on the left in the app there is the ip address)
3. Make sure phone and computer are on the same WiFi network
4. Robot should be running (app shows IP address)
 ### if it is the first time do this(copied from original repo):
      Make sure the system you are running the container on is on the exact same network as the phone (note, public networks like Eduroam won't work.), and observe the IP address shown on the top left of your phone screen. The Robobo UI has a problem where it sometimes cuts off too long IP addresses. You can scan for all active hosts on addresses using nmap (which you need to install as per your operating system). Simply enter all groups you can see, and then scan like so: nmap -sn "192.168.0.*" (Note that nmap is a pen-testing tool and should be used responsibly.)
      Once you have the IP of your phone and are inside the terminal of the docker container, you can run:
      curl http://[Adress shown on top left]:11311
      This should show:
      Empty reply from server 
      Once this is working, you can update the setup.bash file in the scripts directory. Currently, it contains this line:
      export ROS_MASTER_URI=http://localhost:11311
      After this, you should be able to run (again, commands you don't have to understand to run inside the container):
      rosservice call /robot/talk "text: {data: 'Hello'}"


### Run with Hardware
**Terminal 1: Start CoppeliaSim**
```bash
cd /Users/liciatauriello/Desktop/Learning_machine/project/repo_robobo/learning_machines_robobo/examples/full_project_setup
zsh ./scripts/start_coppelia_sim.zsh ./scenes/Robobo_Scene.ttt
```

**Terminal 2: Run with hardware**

```bash
cd /Users/liciatauriello/Desktop/Learning_machine/project/repo_robobo/learning_machines_robobo/examples/full_project_setup
zsh ./scripts/run_apple_sillicon.zsh --hardware
```
---

## Can Simulation and Hardware Run Together?

**No, they cannot.** They are completely separate modes:
- `--simulation` uses CoppeliaSim (virtual robot)
- `--hardware` uses the physical robot via ROS

You can only use one at a time. The code creates either a `SimulationRobobo()` or `HardwareRobobo()` instance based on the flag you provide.

---

## Stopping Everything

### To Stop Simulation:
1. **Terminal 2** (where Docker is running): Press `Ctrl+C`
2. **Terminal 1** (where CoppeliaSim is running): Press `Ctrl+C` or close the CoppeliaSim window

### To Stop Hardware:
- **Terminal**: Press `Ctrl+C`

### Force Stop Docker Containers (if needed):
```bash
docker ps                    # See running containers
docker stop $(docker ps -q)  # Stop all containers
```

## Optional: Create Shortcuts

Add these to your `~/.zshrc`:

```bash
# Robobo shortcuts
alias robobo-sim='cd /Users/liciatauriello/Desktop/Learning_machine/project/repo_robobo/learning_machines_robobo/examples/full_project_setup && zsh ./scripts/start_coppelia_sim.zsh ./scenes/Robobo_Scene.ttt'
alias robobo-run='cd /Users/liciatauriello/Desktop/Learning_machine/project/repo_robobo/learning_machines_robobo/examples/full_project_setup && zsh ./scripts/run_apple_sillicon.zsh'
```

Then reload:
```bash
source ~/.zshrc
```

Now you can use:
- `robobo-sim` to start CoppeliaSim
- `robobo-run --simulation` for simulation
- `robobo-run --hardware` for hardware


