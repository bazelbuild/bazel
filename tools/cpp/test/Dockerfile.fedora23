FROM fedora:23

RUN dnf -y update && dnf clean all
RUN dnf -y install \
    which findutils binutils gcc tar gzip \
    zip unzip java java-devel git clang zlib-devel \
    && dnf clean all
ENV JAVA_HOME /usr/lib/jvm/java-openjdk
