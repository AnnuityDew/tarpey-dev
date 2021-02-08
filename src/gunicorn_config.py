# import native Python packages
import multiprocessing

# Gunicorn config variables
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"
threads = 2
# don't need a timeout on Cloud Run, it has its own
timeout = 0
