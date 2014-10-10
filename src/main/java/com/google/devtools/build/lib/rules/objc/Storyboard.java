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

package com.google.devtools.build.lib.rules.objc;

import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.IBTOOL_PATH;
import static com.google.devtools.build.lib.rules.objc.IosSdkCommands.MINIMUM_OS_VERSION;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.view.actions.CommandLine;
import com.google.devtools.build.xcode.util.Value;

/**
 * Represents a {@code .storyboard} artifact. A storyboard:
 * <ul>
 *   <li>Is a single file with an extension of {@code .storyboard} in its uncompiled, checked-in
 *       form.
 *   <li>Can be in a localized {@code .lproj} directory, including {@code Base.lproj}.
 *   <li>Compiles with {@code ibtool} to a directory with extension {@code .storyboardc} (note the
 *       added "c")
 * </ul>
 * <p>Because {@code ibtool}'s arguments are very similar to {@code actool}, we use
 * {@code actoolzip} to invoke it, with the argument which is usually the path to {@code actool}
 * replaced with the path to {@code ibtool}.
 */
final class Storyboard extends Value<Storyboard> {
  private final Artifact outputZip;
  private final Artifact input;

  Storyboard(Artifact outputZip, Artifact input) {
    super(ImmutableMap.of(
        "outputZip", outputZip,
        "input", input));
    this.outputZip = outputZip;
    this.input = input;
  }

  public Artifact getOutputZip() {
    return outputZip;
  }

  public Artifact getInput() {
    return input;
  }

  /**
   * Generates a set of new instances given the raw storyboard inputs.
   * @param inputs the {@code .storyboard} files.
   * @param intermediateArtifacts the object used to determine the output zip {@link Artifact}s.
   */
  static NestedSet<Storyboard> storyboards(
      Iterable<Artifact> inputs, IntermediateArtifacts intermediateArtifacts) {
    NestedSetBuilder<Storyboard> result = NestedSetBuilder.stableOrder();
    for (Artifact input : inputs) {
      Artifact outputZip = intermediateArtifacts.compiledStoryboardZip(input);
      result.add(new Storyboard(outputZip, input));
    }
    return result.build();
  }

  static ImmutableSet<Artifact> outputZips(Iterable<Storyboard> storyboards) {
    ImmutableSet.Builder<Artifact> outputZips = new ImmutableSet.Builder<>();
    for (Storyboard storyboard : storyboards) {
      outputZips.add(storyboard.getOutputZip());
    }
    return outputZips.build();
  }

  /**
   * Returns the command line that can be used to compile this storyboard and zip the results.
   */
  CommandLine ibtoolzipCommandLine() {
    return new CommandLine() {
      @Override
      public Iterable<String> arguments() {
        return new ImmutableList.Builder<String>()
            // The next three arguments are positional, i.e. they don't have flags before them.
            .add(outputZip.getExecPathString())
            .add(BundleableFile.bundlePath(input) + "c") // archive root
            .add(IBTOOL_PATH)
            .add("--minimum-deployment-target").add(MINIMUM_OS_VERSION)
            .add(input.getExecPathString())
            .build();
      }
    };
  }
}
