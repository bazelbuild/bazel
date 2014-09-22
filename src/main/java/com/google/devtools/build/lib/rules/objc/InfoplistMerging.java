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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.view.FilesToRunProvider;
import com.google.devtools.build.lib.view.actions.ActionConstructionContext;
import com.google.devtools.build.lib.view.actions.CommandLine;
import com.google.devtools.build.lib.view.actions.SpawnAction;
import com.google.devtools.build.xcode.util.Interspersing;

import javax.annotation.Nullable;

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
    @Nullable private NestedSet<Artifact> inputPlists;
    @Nullable private Artifact mergedInfoplist;
    @Nullable private FilesToRunProvider plmerge;

    public Builder(ActionConstructionContext context) {
      this.context = Preconditions.checkNotNull(context);
    }

    public Builder setInputPlists(NestedSet<Artifact> inputPlists) {
      Preconditions.checkState(this.inputPlists == null);
      this.inputPlists = inputPlists;
      return this;
    }

    /**
     * Sets the Artifact corresponding to the merged info plist. Note that this is returned by
     * {@link InfoplistMerging#getPlistWithEverything()} iff there is more than one source.
     */
    public Builder setMergedInfoplist(Artifact mergedInfoplist) {
      Preconditions.checkState(this.mergedInfoplist == null);
      this.mergedInfoplist = mergedInfoplist;
      return this;
    }

    public Builder setPlmerge(FilesToRunProvider plmerge) {
      Preconditions.checkState(this.plmerge == null);
      this.plmerge = plmerge;
      return this;
    }

    /**
     * This static factory method prevents retention of the outer {@link Builder} class reference by
     * the anonymous {@link CommandLine} instance.
     */
    private static CommandLine mergeCommandLine(final Builder builder) {
      return new CommandLine() {
        @Override
        public Iterable<String> arguments() {
          return new ImmutableList.Builder<String>()
              .addAll(Interspersing.beforeEach(
                  "--source_file", Artifact.toExecPaths(builder.inputPlists)))
              .add("--out_file", builder.mergedInfoplist.getExecPathString())
              .build();
        }
      };
    }

    public InfoplistMerging build() {
      Preconditions.checkState(mergedInfoplist != null && plmerge != null,
          "mergedInfoplist (%s) and/or plmerge (%s) is null");
      Action mergeAction = new SpawnAction.Builder(context)
          .setRegisterSpawnAction(false)
          .setMnemonic("Merge Info.plist files")
          .setExecutable(plmerge)
          .setCommandLine(mergeCommandLine(this))
          .addTransitiveInputs(inputPlists)
          .addOutput(mergedInfoplist)
          .build();
      return new InfoplistMerging(
          Iterables.size(inputPlists) == 1
              ? Iterables.getOnlyElement(inputPlists) : mergedInfoplist,
          mergeAction, inputPlists);
    }
  }

  private final Artifact plistWithEverything;
  private final Action mergeAction;
  private final NestedSet<Artifact> inputPlists;

  private InfoplistMerging(
      Artifact plistWithEverything, Action mergeAction, NestedSet<Artifact> inputPlists) {
    this.plistWithEverything = plistWithEverything;
    this.mergeAction = mergeAction;
    this.inputPlists = inputPlists;
  }

  /**
   * Creates action to merge multiple Info.plist files of a binary into a single Info.plist. No
   * action is necessary if there is only one source.
   */
  public Action getMergeAction() {
    return mergeAction;
  }

  public Artifact getPlistWithEverything() {
    return plistWithEverything;
  }

  public NestedSet<Artifact> getInputPlists() {
    return inputPlists;
  }
}
