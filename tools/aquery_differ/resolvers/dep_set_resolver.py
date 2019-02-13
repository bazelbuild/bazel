# Copyright 2018 The Bazel Authors. All rights reserved.
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
"""Contains resolvers for different attributes of Action in aquery output."""
import copy


class DepSetResolver(object):
  """Utility class to resolve the dependency nested set."""

  def __init__(self, dep_set_of_files, artifact_id_to_path):
    self.dep_set_to_artifact_ids = {}
    self.id_to_dep_set = {dep_set.id: dep_set for dep_set in dep_set_of_files}
    self.artifact_id_to_path = artifact_id_to_path

  def resolve(self, dep_set):
    """Given a dep set, return the flattened list of input artifact ids.

    Args:
      dep_set: the dep set object to be resolved.

    Returns:
      The flattened list of input artifact ids.
    """
    if dep_set.id in self.dep_set_to_artifact_ids:
      return self.dep_set_to_artifact_ids[dep_set.id]

    artifact_ids = copy.copy([
        self.artifact_id_to_path[artifact_id]
        for artifact_id in dep_set.direct_artifact_ids
    ])

    for transitive_dep_set_id in dep_set.transitive_dep_set_ids:
      artifact_ids.extend(
          self.resolve(self.id_to_dep_set[transitive_dep_set_id]))

    self.dep_set_to_artifact_ids[dep_set.id] = artifact_ids

    return self.dep_set_to_artifact_ids[dep_set.id]
