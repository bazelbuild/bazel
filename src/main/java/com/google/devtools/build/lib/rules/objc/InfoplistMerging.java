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

import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;

/**
 * Supplies information regarding Infoplist merging for a particular binary. This includes:
 * <ul>
 *   <li>the Info.plist which contains the fields from every source. If there is only one source
 *       plist, this is that plist.
 *   <li>the action to merge all the Infoplists into a single one. This is present even if there is
 *       only one Infoplist, to prevent a Bazel error when an Artifact does not have a generating
 *       action.
 * </ul>
 */
class InfoplistMerging {
  static class Builder {
    private final ActionConstructionContext context;
    private NestedSet<Artifact> inputPlists;
    private FilesToRunProvider plmerge;
    private IntermediateArtifacts intermediateArtifacts;

    public Builder(ActionConstructionContext context) {
      this.context = Preconditions.checkNotNull(context);
    }

    public Builder setInputPlists(NestedSet<Artifact> inputPlists) {
      Preconditions.checkState(this.inputPlists == null);
      this.inputPlists = inputPlists;
      return this;
    }

    public Builder setPlmerge(FilesToRunProvider plmerge) {
      Preconditions.checkState(this.plmerge == null);
      this.plmerge = plmerge;
      return this;
    }

    public Builder setIntermediateArtifacts(IntermediateArtifacts intermediateArtifacts) {
      this.intermediateArtifacts = intermediateArtifacts;
      return this;
    }

    /**
     * This static factory method prevents retention of the outer {@link Builder} class reference by
     * the anonymous {@link CommandLine} instance.
     */
    private static CommandLine mergeCommandLine(
        final NestedSet<Artifact> inputPlists, final Artifact mergedInfoplist) {
      return new CommandLine() {
        @Override
        public Iterable<String> arguments() {
          return new ImmutableList.Builder<String>()
              .addAll(Interspersing.beforeEach(
                  "--source_file", Artifact.toExecPaths(inputPlists)))
              .add("--out_file", mergedInfoplist.getExecPathString())
              .build();
        }
      };
    }

    public InfoplistMerging build() {
      Preconditions.checkNotNull(intermediateArtifacts, "intermediateArtifacts");

      Optional<Artifact> plistWithEverything = Optional.absent();
      Action[] mergeActions = new Action[0];

      int inputs = Iterables.size(inputPlists);
      if (inputs == 1) {
        plistWithEverything = Optional.of(Iterables.getOnlyElement(inputPlists));
      } else if (inputs > 1) {
        Artifact merged = intermediateArtifacts.mergedInfoplist();

        plistWithEverything = Optional.of(merged);
        mergeActions = new SpawnAction.Builder()
            .setMnemonic("MergeInfoPlistFiles")
            .setExecutable(plmerge)
            .setCommandLine(mergeCommandLine(inputPlists, merged))
            .addTransitiveInputs(inputPlists)
            .addOutput(merged)
            .build(context);
      }

      return new InfoplistMerging(plistWithEverything, mergeActions, inputPlists);
    }
  }

  private final Optional<Artifact> plistWithEverything;
  private final Action[] mergeActions;
  private final NestedSet<Artifact> inputPlists;

  private InfoplistMerging(Optional<Artifact> plistWithEverything, Action[] mergeActions,
      NestedSet<Artifact> inputPlists) {
    this.plistWithEverything = plistWithEverything;
    this.mergeActions = mergeActions;
    this.inputPlists = inputPlists;
  }

  /**
   * Creates action to merge multiple Info.plist files of a binary into a single Info.plist. No
   * action is necessary if there is only one source.
   */
  public Action[] getMergeAction() {
    return mergeActions;
  }

  /**
   * An {@link Optional} with the merged infoplist, or {@link Optional#absent()} if there are no
   * merge inputs and it should not be included in the bundle.
   */
  public Optional<Artifact> getPlistWithEverything() {
    return plistWithEverything;
  }

  public NestedSet<Artifact> getInputPlists() {
    return inputPlists;
  }
}
