# Copyright 2026 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wrapper script to execute proguard and then normalize timestamps in the output jar file."""

import argparse
import datetime
import os
import platform
import subprocess
import tempfile
import zipfile

from python.runfiles import Runfiles

_PROGUARD_PATH = "_main/tools/build_defs/proguard/proguard_private"

def lookup_binary(r, path):
    """Lookup the runfiles-adjusted path to a binary.

    Args:
      r: The Runfiles object to use for the lookup.
      path: The path of the binary being found.

    Returns:
      The full path to the binary.

    Raises:
      RuntimeError: If the path is not present in the runfiles, or if the adjusted path does not
      exist on the filesystem.
    """

    if platform.system() == "Windows":
        path = path + ".exe"
    binary = r.Rlocation(path)
    if not binary:
        raise RuntimeError(f"Runfiles failed to resolve {path}")
    elif not os.path.exists(binary):
        raise RuntimeError(f"Runfiles resolved {path} to {binary} but the file does not exist")
    return binary

def apply_proguard(srcs, deps, proguard_spec, output_jar):
    """Call proguard on the given source jars with the spec.

    Args:
      srcs: The source jars to be modified.
      deps: Dependency jars needed to resolve the source jars.
      proguard_spec: The path to the proguard spec file describing what modifications to make.
      output_jar: The path to write the resulting modified jar file to.

    Raises:
      RuntimeError: When the proguard binary fails, includes the stdout and stderr.
    """

    # Set up runfiles and call the proguard binary.
    r = Runfiles.Create()
    proguard_path = lookup_binary(r, _PROGUARD_PATH)

    command = [
        proguard_path,
        "-injars", srcs,
        "-libraryjars", deps,
        "-outjars", output_jar,
        "@" + proguard_spec
    ]

    env = os.environ.copy()
    env.update(r.EnvVars())
    #print("Running proguard: %s" % " ".join(command))
    p = subprocess.run(command, capture_output=True, env=env, check = False)

    if p.returncode != 0:
        message = f"Proguard failed ({p.returncode})"
        stdout = p.stdout.decode()
        if stdout:
            message += f"\n  stdout:\n{stdout}"
        stderr = p.stderr.decode()
        if stderr:
            message += f"\n  stderr:\n{stderr}"
        raise RuntimeError(message)

def reset_timestamps(input_jar, output_jar, timestamp):
    """Rewrite the given jar file to reset all timestamps to a known value.

    Args:
      input_jar: The jar file to be modified.
      output_jar: The path to write the destination jar to.
      timestamp: The known timestamp to modify the output_jar with.
    """
    #print("Resetting timestamps in %s to %s, writing to %s" % (input, timestamp, output))

    with zipfile.ZipFile(input_jar, mode="r") as src:
        with zipfile.ZipFile(output_jar, mode="w") as dest:
            for info in src.infolist():
                #print(f"Filename: {info.filename}")
                #print(f"  Modified: {datetime.datetime(*info.date_time)}")
                data = src.read(info)
                info.date_time = timestamp.timetuple()[:6]
                dest.writestr(info, data)

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resets timestamps in ZIP files", fromfile_prefix_chars="@"
    )
    parser.add_argument(
        "--srcs",
        required=True,
        help="Input jar files, mandatory."
    )
    parser.add_argument(
        "--deps",
        default=[],
        help="Library jar files, optional."
    )
    parser.add_argument(
        "--proguard_spec",
        required=True,
        help="Proguard spec file, mandatory."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="The output file, mandatory."
    )
    parser.add_argument(
        "--timestamp",
        default = "1980-01-01 00:00:00",
        type=datetime.datetime.fromisoformat,
        help = "The timestamp (in ISO format) to set all files to.",
    )
    opts = parser.parse_args()

    with tempfile.TemporaryDirectory() as wdir:
        output_jar = os.path.join(wdir, "stripped.jar")
        apply_proguard(opts.srcs, opts.deps, opts.proguard_spec, output_jar)
        reset_timestamps(output_jar, opts.output, opts.timestamp)


if __name__ == "__main__":
    main()

