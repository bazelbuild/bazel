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
"""ctexplain: how does configuration affect build graphs?

This is a swiss army knife tool that tries to explain why build graphs are the
size they are and how build flags, configuration transitions, and dependency
structures affect that.

This can help developers use flags and transitions with minimal memory and
maximum build speed.

Usage:

  $ ctexplain [--analysis=...] -b "<targets_to_build> [build flags]"

Example:

  $ ctexplain -b "//mypkg:mybinary --define MY_FEATURE=1"

Relevant terms in https://docs.bazel.build/versions/master/glossary.html:
  "target", "configuration", "analysis phase", "configured target",
  "configuration trimming", "transition"

TODO(gregce): link to proper documentation for full details.
"""
from typing import Tuple

from absl import app
from absl import flags

import tools.ctexplain.analyses.summary as summary
import tools.ctexplain.bazel_api as bazel_api
import tools.ctexplain.lib as lib
import tools.ctexplain.util as util

FLAGS = flags.FLAGS

# Available analyses. Key is which --analysis value triggers the analysis, value
# (implementation(cts: Tuple[ConfiguredTarget, ...]), descriptive help text).
analyses = {
    "summary": (
        lambda x: summary.report(summary.analyze(x)),
        "summarizes build graph size and how trimming could help"
    ),
    "culprits": (
        lambda x: print("this analysis not yet implemented"),
        "shows which flags unnecessarily fork configured targets. These\n"
        + "are conceptually mergeable."
    ),
    "forked_targets": (
        lambda x: print("this analysis not yet implemented"),
        "ranks targets by how many configured targets they\n"
        + "create. These may be legitimate forks (because they behave "
        + "differently with\n different flags) or identical clones that are "
        + "conceptually mergeable."
    ),
    "cloned_targets": (
        lambda x: print("this analysis not yet implemented"),
        "ranks targets by how many behavior-identical configured\n targets "
        + "they produce. These are conceptually mergeable."
    )
}


# Command-line flag registration:


def _render_analysis_help_text() -> str:
  """Pretty-prints help text for available analyses."""
  helptext = ""
  for name in analyses:
    helptext += f'- "{name}": {analyses[name][1]}\n'
  return helptext

flags.DEFINE_list("analysis", ["summary"], f"""
Analyses to run. May be any comma-separated combination of

{_render_analysis_help_text()}
""")

flags.register_validator(
    "analysis",
    lambda flag_value: all(name in analyses for name in flag_value),
    message=f'available analyses: {", ".join(analyses.keys())}')

flags.DEFINE_multi_string(
    "build", [],
    """command-line invocation of the build to analyze. For example:
"//foo --define a=b". If listed multiple times, this is a "multi-build
analysis" that measures how much distinct builds can share subgraphs""",
    short_name="b")


# Core program logic:


def _get_build_flags(cmdline: str) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
  """Parses a build invocation command line.

  Args:
    cmdline: raw build invocation string. For example: "//foo --cpu=x86"

  Returns:
    Tuple of ((target labels to build), (build flags))
  """
  cmdlist = cmdline.split()
  labels = [arg for arg in cmdlist if arg.startswith("//")]
  build_flags = [arg for arg in cmdlist if not arg.startswith("//")]
  return (tuple(labels), tuple(build_flags))


def main(argv):
  del argv  # Satisfy py linter's "unused" warning.
  if not FLAGS.build:
    exit("ctexplain: build efficiency measurement tool. Add --help "
         + "for usage.")
  elif len(FLAGS.build) > 1:
    exit("TODO(gregce): support multi-build shareability analysis")

  (labels, build_flags) = _get_build_flags(FLAGS.build[0])
  build_desc = ",".join(labels)
  with util.ProgressStep(f"Collecting configured targets for {build_desc}"):
    cts = lib.analyze_build(bazel_api.BazelApi(), labels, build_flags)
  for analysis in FLAGS.analysis:
    analyses[analysis][0](cts)


if __name__ == "__main__":
  app.run(main)
