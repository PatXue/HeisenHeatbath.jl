import Pkg
Pkg.activate("..")

using HeisenHeatbath
using Carlo
using Carlo.JobTools

tm = TaskMaker()
tm.rand_init = true

L = 20
tm.Lx = tm.Ly = L
tm.sweeps = 1000L
tm.thermalization = 100L
tm.binsize = 10L

tm.J = 1.0
tm.H = 0.0
Ts = 0.1:0.1:4.0
for T in Ts
    tm.T = T
    task(tm)
end

job = JobInfo("test-sweep", HeisenHeatbathMC;
    run_time = "24:00:00",
    checkpoint_time = "30:00",
    tasks = make_tasks(tm),
)
start(job, ARGS)