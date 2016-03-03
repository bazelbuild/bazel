# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A simple test runner for docker (experimental)."""

import copy
import os
import os.path
import shlex
import StringIO
import subprocess
import sys
import threading

from third_party.py import gflags

gflags.DEFINE_multistring(
    "image", [],
    "The list of additional docker image to load (path to a docker_build "
    "target), optional.")

gflags.DEFINE_string(
    "main", None,
    "The main image to run (path to a docker_build target), mandatory.")
gflags.MarkFlagAsRequired("main")

gflags.DEFINE_string(
    "cmd", None,
    "A command to provide to the docker image, optional (default: use the "
    "entrypoint).")

gflags.DEFINE_string("docker", "docker", "Path to the docker client binary.")

gflags.DEFINE_boolean("verbose", True, "Be verbose.")

FLAGS = gflags.FLAGS

LOCAL_IMAGE_PREFIX = "bazel/docker_test:"


def _copy_stream(in_stream, out_stream):
  for c in iter(lambda: in_stream.read(1), ""):
    out_stream.write(c)
    out_stream.flush()


def execute(command, stdout=sys.stdout, stderr=sys.stderr, env=None):
  """Execute a command while redirecting its output streams."""
  if FLAGS.verbose:
    print "Executing '%s'" % " ".join(command)
  p = subprocess.Popen(command,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       env=env)
  t1 = threading.Thread(target=_copy_stream, args=[p.stdout, stdout])
  t2 = threading.Thread(target=_copy_stream, args=[p.stderr, stderr])
  t1.daemon = True
  t2.daemon = True
  t1.start()
  t2.start()
  p.wait()
  t1.join()
  t2.join()
  return p.returncode


def load_image(image):
  """Load a docker image using the runner provided by docker_build."""
  tag = LOCAL_IMAGE_PREFIX + image.replace("/", "_")
  err = StringIO.StringIO()
  env = copy.deepcopy(os.environ)
  env["DOCKER"] = FLAGS.docker
  ret = execute([image, tag], stderr=err, env=env)
  if ret != 0:
    sys.stderr.write("Error loading image %s (return code: %s):\n" %
                     (image, ret))
    sys.stderr.write(err.getvalue())
    return None
  return tag


def load_images(images):
  """Load a series of docker images using the docker_build's runner."""
  print "### Image loading ###"
  return [load_image(image) for image in images]


def cleanup_images(tags):
  """Remove docker tags and images previously loaded."""
  print "### Image cleanup ###"
  for tag in tags:
    if isinstance(tag, basestring):
      execute([FLAGS.docker, "rmi", tag])


def run_image(tag):
  """Run a docker image, in background."""
  print "Running " + tag
  out = StringIO.StringIO()
  err = StringIO.StringIO()
  process = execute([FLAGS.docker, "run", "--rm", tag], out, err)
  if process.wait() != 0:
    sys.stderr.write("Error running docker run on %s:\n" % tag)
    sys.stderr.write(err.getvalue())
    return None
  else:
    return out.getvalue().strip()


def run_images(tags):
  """Run a list of docker images, in background."""
  print "### Running images ###"
  return [run_image(tag) for tag in tags]


def cleanup_containers(containers):
  """Kill containers."""
  print "### Containers cleanup ###"
  for c in containers:
    if isinstance(c, basestring):
      execute([FLAGS.docker, "kill", c])


def main(unused_argv):
  tags = load_images([FLAGS.main] + FLAGS.image)
  if None in tags:
    cleanup_images(tags)
    return -1
  try:
    containers = run_images(tags[1:])
    ret = -1
    if None not in containers:
      print "### Running main container ###"
      ret = execute([
          FLAGS.docker,
          "run",
          "--rm",
          "--attach=STDOUT",
          "--attach=STDERR", tags[0]
          ] + ([] if FLAGS.cmd is None else shlex.split(FLAGS.cmd)))
  finally:
    cleanup_containers(containers)
    cleanup_images(tags)
  return ret


if __name__ == "__main__":
  sys.exit(main(FLAGS(sys.argv)))
