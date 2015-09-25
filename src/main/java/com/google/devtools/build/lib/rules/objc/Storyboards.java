// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.objc;


import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;

/**
 * Contains information about storyboards for a single target. This does not include information
 * about the transitive closure. A storyboard:
 * <ul>
 *   <li>Is a single file with an extension of {@code .storyboard} in its uncompiled, checked-in
 *       form.
 *   <li>Can be in a localized {@code .lproj} directory, including {@code Base.lproj}.
 *   <li>Compiles with {@code ibtool} to a directory with extension {@code .storyboardc} (note the
 *       added "c")
 * </ul>
 *
 * <p>The {@link NestedSet}s stored in this class are only one level deep, and do not include the
 * storyboards in the transitive closure. This is to facilitate structural sharing between copies
 * of the sequences - the output zips can be added transitively to the inputs of the merge bundle
 * action, as well as to the files to build set, and only one instance of the sequence exists for
 * each set.
 */
final class Storyboards {
  private final NestedSet<Artifact> outputZips;
  private final NestedSet<Artifact> inputs;

  private Storyboards(NestedSet<Artifact> outputZips, NestedSet<Artifact> inputs) {
    this.outputZips = outputZips;
    this.inputs = inputs;
  }

  public NestedSet<Artifact> getOutputZips() {
    return outputZips;
  }

  public NestedSet<Artifact> getInputs() {
    return inputs;
  }

  static Storyboards empty() {
    return new Storyboards(
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER));
  }

  /**
   * Generates a set of new instances given the raw storyboard inputs.
   * @param inputs the {@code .storyboard} files.
   * @param intermediateArtifacts the object used to determine the output zip {@link Artifact}s.
   */
  static Storyboards fromInputs(
      Iterable<Artifact> inputs, IntermediateArtifacts intermediateArtifacts) {
    NestedSetBuilder<Artifact> outputZips = NestedSetBuilder.stableOrder();
    for (Artifact input : inputs) {
      outputZips.add(intermediateArtifacts.compiledStoryboardZip(input));
    }
    return new Storyboards(outputZips.build(), NestedSetBuilder.wrap(Order.STABLE_ORDER, inputs));
  }
}
