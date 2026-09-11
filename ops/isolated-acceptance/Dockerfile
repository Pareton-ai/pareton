FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN sed -i 's|//archive.ubuntu.com|//mirrors.aliyun.com|g; s|//security.ubuntu.com|//mirrors.aliyun.com|g' /etc/apt/sources.list.d/ubuntu.sources 2>/dev/null; \
    apt-get update && apt-get install -y --no-install-recommends \
    systemd systemd-sysv python3 ca-certificates git && \
    apt-get clean && rm -rf /var/lib/apt/lists/*
COPY vector /usr/bin/vector
RUN chmod 0755 /usr/bin/vector && vector --version
# Trim systemd for container life and keep a persistent journal for A19.
RUN cd /lib/systemd/system/sysinit.target.wants/ && \
    ls | grep -v systemd-tmpfiles-setup | xargs rm -f || true; \
    rm -f /lib/systemd/system/multi-user.target.wants/* /etc/systemd/system/*.gettytarget 2>/dev/null || true; \
    systemctl set-default multi-user.target; \
    mkdir -p /var/log/journal /etc/systemd/system/systemd-journald.service.d && \
    printf '[Journal]\nStorage=persistent\n' \
      > /etc/systemd/system/systemd-journald.service.d/storage.conf
CMD ["/sbin/init"]
