import os
import random
import resource

from flask import Flask

app = Flask(__name__)

DELAY = float(os.environ.get('DELAY', '1'))

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def busy_wait(path):
    usage_start = resource.getrusage(resource.RUSAGE_THREAD)
    user_start_time = usage_start.ru_utime
    system_start_time = usage_start.ru_stime

    k = 3
    cpu_time = sum(random.expovariate(k / DELAY) for _ in range(k))
    #cpu_time = DELAY

    while True:
        usage_current = resource.getrusage(resource.RUSAGE_THREAD)
        user_current_time = usage_current.ru_utime
        system_current_time = usage_current.ru_stime

        elapsed_time = (user_current_time + system_current_time) - (user_start_time + system_start_time)

        if elapsed_time >= cpu_time:
            break

    user_time_diff = user_current_time - user_start_time
    system_time_diff = system_current_time - system_start_time

    return f"{user_time_diff},{system_time_diff}\n"

if __name__ == '__main__':
    app.run(
        threaded=False
    )