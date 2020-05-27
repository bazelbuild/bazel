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
"""The core data types ctexplain manipulates."""

from typing import Dict
from typing import Optional
from typing import Tuple
from dataclasses import dataclass
from dataclasses import field


@dataclass(frozen=True)
class Configuration():
  """Stores a build configuration as a collection of fragments and options."""
  # BuildConfiguration.Fragments in this configuration, as base names without
  # packages. For example: ["PlatformConfiguration", ...].
  fragments: Tuple[str, ...]
  # Dict of FragmentOptions to option key/value pairs. For example:
  # {"CoreOptions": {"action_env": "[]", "cpu": "x86", ...}, ...}.
  #
  # Option values are stored as strings of whatever "bazel config" outputs.
  #
  # Note that Fragment and FragmentOptions aren't the same thing.
  options: Dict[str, Dict[str, str]]

  def __hash__(self):
    sorted_fragment_options = sorted(self.options.keys())
    items_to_hash = [self.fragments, (tuple(sorted_fragment_options))]
    for fragment_option in sorted_fragment_options:
      items_to_hash.append(tuple(sorted(self.options[fragment_option])))
    return hash(tuple(items_to_hash))


@dataclass(frozen=True)
class ConfiguredTarget():
  """Encapsulates a target + configuration + required fragments."""
  # Label of the target this represents.
  label: str
  # Configuration this target is applied to. May be None.
  config: Optional[Configuration]
  # The hash of this configuration as reported by Bazel.
  config_hash: str
  # Fragments required by this configured target and its transitive
  # dependencies. Stored as base names without packages. For example:
  # "PlatformOptions".
  transitive_fragments: Tuple[str, ...]


@dataclass(frozen=True)
class HostConfiguration(Configuration):
  """Special marker for the host configuration.

  There's exactly one host configuration per build, so we shouldn't suggest
  merging it with other configurations.

  TODO(gregce): suggest host configuration trimming once we figure out the right
  criteria. Even if Bazel's not technically equipped to do the trimming, it's
  still theoretically valuable information. Note that moving from host to exec
  configurations make this all a little less relevant, since exec configurations
  aren't "special" compared to normal configurations.
  """
  # We don't currently read the host config's fragments or option values.
  fragments: Tuple[str, ...] = ()
  options: Dict[str, Dict[str, str]] = field(default_factory=lambda: {})


@dataclass(frozen=True)
class NullConfiguration(Configuration):
  """Special marker for the null configuration.

  By definition this has no fragments or options.
  """
  fragments: Tuple[str, ...] = ()
  options: Dict[str, Dict[str, str]] = field(default_factory=lambda: {})
