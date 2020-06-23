all:
	sudo apt-get install build-essential openjdk-11-jdk python zip unzip
	env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk" bash ./compile.sh
clean:
	rm -rf output/bazel
