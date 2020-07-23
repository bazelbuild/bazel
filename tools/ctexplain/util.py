# Lint as: python3
# Copyright 2020 The Bazel Authors. All rights reserved.
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
"""Generic utilities."""
import time

class ProgressStep:
  """A simple context manager that prints a progress message.

    Forked from a similar project by brandjon@google.com.
  """

  def __init__(self, msg, show_done=True):
    self.msg = msg
    self.show_done = show_done

  def __enter__(self):
    print(self.msg + "...", flush=True, end="")
    self.start_time = time.perf_counter()

  def __exit__(self, exc_type, exc_value, traceback):
    if self.show_done:
      elapsed_sec = time.perf_counter() - self.start_time
      if elapsed_sec < 0.1:
        time_str = f"{elapsed_sec * 1000:.0f} ms"
      else:
        time_str = f"{elapsed_sec:.2f} s"
      print(f" done in {time_str}.", flush=True)


def percent_diff(val1, val2):
  """Returns what percentage a change val2 is from val1.

  For example, if val=10 and val2=15, returns '+50%'.

  Args:
    val1: Base number.
    val2: Number to compare against the base number.
  """
  diff = (val2 - val1) / val1 * 100
  return f"+{diff:.1f}%" if diff >= 0 else f"{diff:.1f}%"