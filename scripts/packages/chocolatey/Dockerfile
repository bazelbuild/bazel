FROM linuturk/mono-choco

ARG PACKAGE=bazel
ARG PACKAGE_SOURCE_PATH=bazel-package
ARG VERSION=0.11.0-rc2
ARG API_KEY=replace_me_with_real_key

WORKDIR /opt/choco

# Copy our package files into the container
COPY ${PACKAGE_SOURCE_PATH} /opt/choco

# Check the files that were copied
RUN ls -R /opt/choco

# Check our version
RUN choco --version

# Pack the nupkg
RUN choco pack /opt/choco/${PACKAGE}.nuspec --outputdirectory /tmp

RUN ls -R /tmp

# Example push command
RUN choco push --verbose --debug --timeout 30 /tmp/${PACKAGE}.${VERSION}.nupkg --apikey "${API_KEY}"
