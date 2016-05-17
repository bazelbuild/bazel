# Copyright 2016 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http:#www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Small testing framework for Skylark rules and aspects.

def start_test(ctx):
  return struct(errors = [], ctx = ctx)

def end_test(env):
  cmd = "\n".join([
      "cat << EOF",
      "\n".join(reversed(env.errors)),
      "EOF",
      "exit %d" % (1 if env.errors else 0)])
  env.ctx.file_action(
          output = env.ctx.outputs.executable,
          content = cmd,
          executable = True,
  )

def fail_test(env, msg):
  print(msg)
  env.errors.append(msg)


def assert_equals(env, expected, actual):
  if not expected == actual:
    fail_test(env, "'%s' != '%s'" % (expected, actual))

def assert_true(env, condition, message):
  if not condition:
    fail_test(env, message)
