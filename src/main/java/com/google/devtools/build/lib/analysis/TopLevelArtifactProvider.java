// Copyright 2014 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.analysis;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * {@code ConfiguredTarget}s implementing this interface can provide artifacts that <b>can</b> be
 * built when the target is mentioned on the command line (as opposed to being always built, like
 * {@link com.google.devtools.build.lib.analysis.FileProvider})
 *
 * <p>The artifacts are grouped into "output groups". Which output groups are built is controlled
 * by the {@code --output_groups} undocumented command line option, which in turn is added to the
 * command line at the discretion of the build command being run.
 *
 * <p>Output groups starting with an underscore are "not important". This means that artifacts built
 * because such an output group is mentioned in a {@code --output_groups} command line option are
 * not mentioned on the output.
 */
@Immutable
public final class TopLevelArtifactProvider implements TransitiveInfoProvider {
  public static final String HIDDEN_OUTPUT_GROUP_PREFIX = "_";
  public static final String BASELINE_COVERAGE = HIDDEN_OUTPUT_GROUP_PREFIX + "_baseline_coverage";
  public static final String FILES_TO_COMPILE = "files_to_compile";
  public static final String COMPILATION_PREREQUISITES = "compilation_prerequisites";

  private final ImmutableMap<String, NestedSet<Artifact>> outputGroups;

  TopLevelArtifactProvider(ImmutableMap<String, NestedSet<Artifact>> outputGroups) {
    this.outputGroups = outputGroups;
  }

  /** Return the artifacts in a particular output group.
   *
   * @return the artifacts in the output group with the given name. The return value is never null.
   *     If the specified output group is not present, the empty set is returned.
   */
  public NestedSet<Artifact> getOutputGroup(String outputGroupName) {
    return outputGroups.containsKey(outputGroupName)
        ? outputGroups.get(outputGroupName)
        : NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER);
  }
}
