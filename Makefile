# Copyright 2014 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This script bootstraps building a Bazel binary without Bazel then
# use this compiled Bazel to bootstrap Bazel itself. It can also
# be provided with a previous version of Bazel to bootstrap Bazel itself.
#
# The resulting binary can be found at output/bazel.

all:
	# Make sure dependencies for bootstrapping bazel are installed
	sudo apt-get install build-essential openjdk-11-jdk python zip unzip
	env EXTRA_BAZEL_ARGS="--host_javabase=@local_jdk//:jdk"
	
	# Set the default verbose mode in buildenv.sh so that we do not display command
	# output unless there is a failure.  We do this conditionally to offer the user
	# a chance of overriding this in case they want to do so.
	env VERBOSE=no
	
	# Set up the environment for building and output
	set -o errexit
	cd "$(dirname "$0")"
	source scripts/bootstrap/buildenv.sh
	source scripts/bootstrap/bootstrap.sh
	mkdir -p output
	env BAZEL=
	
	# Create an initial binary so we can host ourself
	if [ ! -x "${BAZEL}" ]; then
  		new_step 'Building Bazel from scratch'
  		source scripts/bootstrap/compile.sh
	fi
	
	# Bootstrap bazel using the previous bazel binary = release binary
	if [ "${EMBED_LABEL-x}" = "x" ]; then
  		# Add a default label when unspecified
  		git_sha1=$(git_sha1)
  		EMBED_LABEL="$(get_last_version) (@${git_sha1:-non-git})"
	fi

	# Set host and target platform directly because we are building for the local host
	if [[ $PLATFORM == "darwin" ]] && \
    		xcodebuild -showsdks 2> /dev/null | grep -q '\-sdk iphonesimulator'; then
  		EXTRA_BAZEL_ARGS="${EXTRA_BAZEL_ARGS-} --define IPHONE_SDK=1"
	fi

	# Commence Bootstrapping Bazel
	new_step 'Building Bazel with Bazel'
	display "."
	log "Building output/bazel"
	bazel_build "src:bazel_nojdk${EXE_EXT}" \
  		--action_env=PATH \
  		--host_platform=@local_config_platform//:host \
  		--platforms=@local_config_platform//:host \
  		|| fail "Could not build Bazel"
		bazel_bin_path="$(get_bazel_bin_path)/src/bazel_nojdk${EXE_EXT}"
		[ -e "$bazel_bin_path" ] \
  		|| fail "Could not find freshly built Bazel binary at '$bazel_bin_path'"
		cp -f "$bazel_bin_path" "output/bazel${EXE_EXT}" \
  		|| fail "Could not copy '$bazel_bin_path' to 'output/bazel${EXE_EXT}'"
	
	# Set Permissions for the new binary in the output directory
	chmod 0755 "output/bazel${EXE_EXT}"
	BAZEL="$(pwd)/output/bazel${EXE_EXT}"
	clear_log
	display "Build successful! Binary is here: ${BAZEL}"
	
clean:
	# Prompt the user to make sure they want to remove the output directory
	read -p "Are you sure you want to remove the Bazel Bootstrap output? [Y/n]: " answer
	if [[ $answer == "Y" ]] || [[ $answer == "y" ]]; then
		display "Removing Bazel Bootstrap output..."
		rm -rf output/bazel
		display "Done!"
	else
		display "Nothing to do."
	fi
