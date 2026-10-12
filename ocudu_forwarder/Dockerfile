FROM ocudu-tools:joint-local

COPY requirements.txt /opt/ocudu-forwarder/requirements.txt
RUN python -m pip install --no-cache-dir -r /opt/ocudu-forwarder/requirements.txt

COPY forward.py /opt/ocudu-forwarder/forward.py

ENTRYPOINT ["python", "-u", "/opt/ocudu-forwarder/forward.py"]
