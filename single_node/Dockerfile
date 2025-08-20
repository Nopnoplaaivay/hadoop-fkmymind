FROM ubuntu:20.04

# Cài đặt các dependencies
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk wget ssh openssh-server openssh-client rsync sudo && \
    apt-get clean

# Tạo user hadoop
RUN useradd -m -s /bin/bash hadoop && \
    echo 'hadoop:hadoop' | chpasswd && \
    adduser hadoop sudo

# Cài đặt Hadoop
USER hadoop
WORKDIR /home/hadoop

RUN wget https://archive.apache.org/dist/hadoop/common/hadoop-3.3.4/hadoop-3.3.4.tar.gz && \
    tar -xzf hadoop-3.3.4.tar.gz && \
    mv hadoop-3.3.4 hadoop && \
    rm hadoop-3.3.4.tar.gz

# Thiết lập biến môi trường
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
ENV HADOOP_HOME=/home/hadoop/hadoop
ENV PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin
ENV HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop

# Thêm environment variables vào bash profile để SSH session có thể sử dụng
RUN echo 'export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64' >> /home/hadoop/.bashrc && \
    echo 'export HADOOP_HOME=/home/hadoop/hadoop' >> /home/hadoop/.bashrc && \
    echo 'export PATH=$PATH:$HADOOP_HOME/bin:$HADOOP_HOME/sbin' >> /home/hadoop/.bashrc && \
    echo 'export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop' >> /home/hadoop/.bashrc

# Copy các file cấu hình và startup script
USER root
COPY config/* /home/hadoop/hadoop/etc/hadoop/
COPY start-hadoop.sh /home/hadoop/
COPY check-hadoop.sh /home/hadoop/

# cấp quyền chạy và đổi quyền chủ sở hữu
RUN chown -R hadoop:hadoop /home/hadoop/hadoop && \
    # chmod +x /home/hadoop/hadoop/etc/hadoop/hadoop-env.sh && \
    chmod +x /home/hadoop/start-hadoop.sh && \
    chmod +x /home/hadoop/check-hadoop.sh && \
    chown hadoop:hadoop /home/hadoop/start-hadoop.sh && \
    chown hadoop:hadoop /home/hadoop/check-hadoop.sh

# SSH configuration for root
RUN mkdir -p ~/.ssh && \
    ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 0600 ~/.ssh/authorized_keys && \
    echo "Host *" >> ~/.ssh/config && \
    echo "  StrictHostKeyChecking no" >> ~/.ssh/config && \
    echo "  UserKnownHostsFile=/dev/null" >> ~/.ssh/config && \
    chmod 600 ~/.ssh/config

# SSH configuration for hadoop user
USER hadoop
RUN mkdir -p ~/.ssh && \
    ssh-keygen -t rsa -P '' -f ~/.ssh/id_rsa && \
    cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys && \
    chmod 0600 ~/.ssh/authorized_keys && \
    echo "Host *" >> ~/.ssh/config && \
    echo "  StrictHostKeyChecking no" >> ~/.ssh/config && \
    echo "  UserKnownHostsFile=/dev/null" >> ~/.ssh/config && \
    chmod 600 ~/.ssh/config

# Expose ports
EXPOSE 9870 8088 9000

# Switch to hadoop user and set working directory
USER hadoop
WORKDIR /home/hadoop

CMD ["./start-hadoop.sh"]