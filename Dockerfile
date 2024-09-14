FROM python:3.12-slim

WORKDIR /Sine_gates

COPY docker_reqs/ docker_reqs/
COPY reqs.txt reqs.txt

RUN pip install --no-cache-dir -r reqs.txt

RUN chmod +x docker_reqs/run_loop.sh
CMD ["./docker_reqs/run_loop.sh"]
