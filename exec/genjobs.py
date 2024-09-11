from src.analysis.spawnjobs import JobLoader, pjoin, rs
import os

cwd = os.path.dirname(os.path.abspath(__file__))

def gen_jobs():
    jl = JobLoader(jobpath=pjoin(cwd, rs.JOB_DIRNAME))
    jl.writejobs()

if __name__ == '__main__':
    gen_jobs()
