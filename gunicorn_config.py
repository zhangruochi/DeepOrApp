import os

if "GUNICORN_WORKER_NUM" in os.environ:
    GUNICORN_WORKER_NUM =  os.environ["GUNICORN_WORKER_NUM"]
else:
    GUNICORN_WORKER_NUM = 4
       
debug = False
loglevel = 'info'
bind = "0.0.0.0:5532"
timeout = 600  #超时

daemon = False  # 意味着开启后台运行，默认为False
workers = GUNICORN_WORKER_NUM  # 启动的进程数
# workers = 8
threads = 1  #指定每个进程开启的线程数
worker_connections = 100
worker_class = 'uvicorn.workers.UvicornWorker'  #默认为sync模式，也可使用gevent模式。
x_forwarded_for_header = 'X-FORWARDED-FOR'