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
"""General-purpose business logic."""
from typing import List
from typing import Mapping
from typing import Set
from typing import Tuple

from frozendict import frozendict
from tools.ctexplain.bazel_api import BazelApi
from tools.ctexplain.types import Configuration
from tools.ctexplain.types import ConfiguredTarget


def analyze_build(bazel: BazelApi, labels: Tuple[str, ...],
                  build_flags: Tuple[str, ...]) -> Tuple[ConfiguredTarget, ...]:
  """Gets a build invocation's configured targets.

  Args:
    bazel: API for invoking Bazel.
    labels: The targets to build.
    build_flags: The build flags to use.

  Returns:
    Configured targets representing the build.

  Raises:
    RuntimeError: On any invocation errors.
  """
  cquery_args = [f'deps({",".join(labels)})']
  cquery_args.extend(build_flags)
  (success, stderr, cts) = bazel.cquery(cquery_args)
  if not success:
    raise RuntimeError("invocation failed: " + stderr.decode("utf-8"))

  # We have to do separate calls to "bazel config" to get the actual configs
  # from their hashes.
  hashes_to_configs = {}
  cts_with_configs = []
  for ct in cts:
    # Don't use dict.setdefault because that unconditionally calls get_config
    # as one of its parameters and that's an expensive operation to waste.
    if ct.config_hash not in hashes_to_configs:
      hashes_to_configs[ct.config_hash] = bazel.get_config(ct.config_hash)
    config = hashes_to_configs[ct.config_hash]
    cts_with_configs.append(
        ConfiguredTarget(
            ct.label,
            config,
            ct.config_hash,
            ct.transitive_fragments))

  return tuple(cts_with_configs)


def trim_configured_targets(
    cts: Tuple[ConfiguredTarget, ...]
    ) -> Mapping[ConfiguredTarget, Tuple[ConfiguredTarget, ...]]:
  """Trims a set of configured targets to only include needed fragments.

  An "untrimmed" configured target contains all fragments and user-defined
  options (like Starlark flags or --define) in its configuration, regardless of
  whether the target actually needs them. For example, configurations include
  both Java-related and C++-related fragments that encapsulate Java-related
  flags and C++ flags, respectively. A C++ binary doesn't "need" the Java
  fragment since C++ compilation doesn't use Java flags.

  A "trimmed" configured target only includes the fragments and user-defined
  options actually needed. So a trimmed cc_binary includes the C++ fragment but
  not the Java fragment. It includes --define foo=bar if it has a select() on
  --foo (or the "$(foo)" Make variable in one of its attributes), otherwise it
  drops it. And so on.

  Args:
    cts: Set of untrimmed configured targets.

  Returns:
    A mapping from each trimmed configured target to the untrimmed ones that
    that reduce to it. For example, if a cc_binary appears in two configurations
    that only differ in Java options, the trimmed, the map has one key (the
    config minus its java options) with two values (each original config).
  """
  ans = {}
  for ct in cts:
    trimmed_ct = _trim_configured_target(ct, _get_required_fragments(ct))
    ans.setdefault(trimmed_ct, []).append(ct)
  return ans


# All configurations have the same config fragment -> options fragment map.
# Compute it once for efficiency.
_options_to_fragments: Mapping[str, List[str]] = {}


def _get_required_fragments(ct: ConfiguredTarget) -> Set[str]:
  """Normalizes and returns the fragments required by a configured target.

  Normalization means:
    - If a rule class (like cc_binary) requires a fragment, include it.
    - If a select() requires a flag, figure out which options fragment (which
      is not the same as a configuration fragment) that means. For example,
      "--copt" belongs to options fragment CppOptions. Then choose one of the
      configuration fragments that requires CppOptions: CppConfiguration or
      ObjcConfiguration. Either is sufficient to guarantee the presence of
      CppOptions. Heuristically, we choose the fragment requiring the smallest
      number of option fragments to try to be as granular as possible.
    - If a target consumes --define foo=<val> anywhere, include "--define foo".
    - If a target consumes a Starlark flag, include that flag's label.

  Args:
    ct: Untrimmed configured target.

  Returns:
    Set of required configuration pieces.
  """
  if not ct.config.fragments:  # Null configurations are already empty.
    return set()
  if not _options_to_fragments:
    for fragment, options_fragments in ct.config.fragments.items():
      for o in options_fragments:
        _options_to_fragments.setdefault(o, set()).add(fragment)

  ans = []
  for req in ct.transitive_fragments:
    if (req in ct.config.fragments or req.startswith("--") or
        req == "CoreOptions"):
      # The requirement is a fragment, Starlark flag, or core (universally
      # required) option.
      ans.append(req)
    elif req in _options_to_fragments:
      # The requirement is a native option (like "--copt"). Find a matching
      # fragment.
      fragments = _options_to_fragments[req]
      matches = [(len(ct.config.fragments[frag]), frag) for frag in fragments]
      # TODO(gregce): scan for existing fulfilling fragment, prefer that.
      ans.append(sorted(matches)[0][1])  # Sort each entry by count, then name.
    else:
      raise ValueError(f"{ct.label}: don't understand requirement {req}")
  return set(ans)


# CoreOptions entries that should always be trimmed because they change with
# configuration changes but don't actually cause those changes.
_trimmable_core_options = (
    "affected by starlark transition",
    "transition directory name fragment",
)


def _trim_configured_target(ct: ConfiguredTarget,
                            required_fragments: Set[str]) -> ConfiguredTarget:
  """Trims a configured target to only the config pieces it needs.

  Args:
    ct: Untrimmed configured tareget.
    required_fragments: Configuration pieces the configured target requires,
      as reported by _get_required_fragments.

  Returns:
    Trimmed copy of the configured target. Both config.fragments and
    config.options are suitably trimmed.
  """
  trimmed_options = {}
  for (options_class, options) in ct.config.options.items():
    # CoreOptions are universally included with no owning fragments.
    if options_class == "CoreOptions":
      trimmed_options[options_class] = frozendict({
          k: v for k, v in options.items() if k not in _trimmable_core_options
      })
    elif options_class == "user-defined":
      # Include each user-defined option on a case-by-case basis, since
      # user-defined requirements are declared directly on each option.
      trimmed_options["user-defined"] = frozendict({
          name: val
          for name, val in ct.config.options["user-defined"].items()
          if name in required_fragments
      })
    else:
      associated_fragments = set(_options_to_fragments[options_class])
      if associated_fragments & required_fragments:
        trimmed_options[options_class] = options

  trimmed_fragments = frozendict({
      fragment: options for fragment, options in ct.config.fragments.items()
      if fragment in required_fragments
  })
  trimmed_config = Configuration(trimmed_fragments, frozendict(trimmed_options))
  return ConfiguredTarget(
      ct.label, trimmed_config, "trimmed hash", ct.transitive_fragments)
