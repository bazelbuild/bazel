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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.actions.ActionConstructionContext;
import com.google.devtools.build.lib.analysis.actions.CommandLine;
import com.google.devtools.build.lib.analysis.actions.CustomCommandLine;
import com.google.devtools.build.lib.analysis.actions.SpawnAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;

/**
 * Supplies information regarding Infoplist merging for a particular binary. This includes:
 * <ul>
 *   <li>the Info.plist which contains the fields from every source. If there is only one source
 *       plist, this is that plist.
 *   <li>the action to merge all the Infoplists into a single one and stamp the bundle ID on it.
 *       This action is present if there is more than one plist file or there is a non-null bundle
 *       ID to stamp on the merged plist file.
 * </ul>
 */
class InfoplistMerging {
  static class Builder {
    private final ActionConstructionContext context;
    private NestedSet<Artifact> inputPlists;
    private FilesToRunProvider plmerge;
    private IntermediateArtifacts intermediateArtifacts;
    private String primaryBundleId;
    private String fallbackBundleId;

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
     * Sets the potential bundle identifiers to stamp on the merged plist file.
     *
     * @param primaryBundleId used to set the bundle identifier or override the existing one from
     *     plist file, can be null
     * @param fallbackBundleId used to set the bundle identifier if it is not set by plist file or
     *     primary identifier, can be null
     */
    public Builder setBundleIdentifiers(String primaryBundleId, String fallbackBundleId) {
      this.primaryBundleId = primaryBundleId;
      this.fallbackBundleId = fallbackBundleId;
      return this;
    }

    /**
     * This static factory method prevents retention of the outer {@link Builder} class reference by
     * the anonymous {@link CommandLine} instance.
     */
    private static CommandLine mergeCommandLine(NestedSet<Artifact> inputPlists,
        Artifact mergedInfoplist, String primaryBundleId,  String fallbackBundleId) {
      CustomCommandLine.Builder argBuilder = CustomCommandLine.builder()
          .addBeforeEachExecPath("--source_file", inputPlists)
          .addExecPath("--out_file", mergedInfoplist);

      if (primaryBundleId != null) {
        argBuilder.add("--primary_bundle_id").add(primaryBundleId);
      }

      if (fallbackBundleId != null) {
        argBuilder.add("--fallback_bundle_id").add(fallbackBundleId);
      }

      return argBuilder.build();
    }

    public InfoplistMerging build() {
      Preconditions.checkNotNull(intermediateArtifacts, "intermediateArtifacts");

      Optional<Artifact> plistWithEverything = Optional.absent();
      Action[] mergeActions = new Action[0];

      if (!inputPlists.isEmpty()) {
        int inputs = Iterables.size(inputPlists);
        if (inputs == 1 && primaryBundleId == null && fallbackBundleId == null) {
          plistWithEverything = Optional.of(Iterables.getOnlyElement(inputPlists));
        } else {
          Artifact merged = intermediateArtifacts.mergedInfoplist();

          plistWithEverything = Optional.of(merged);
          mergeActions = new SpawnAction.Builder()
              .setMnemonic("MergeInfoPlistFiles")
              .setExecutable(plmerge)
              .setCommandLine(
                  mergeCommandLine(inputPlists, merged, primaryBundleId, fallbackBundleId))
              .addTransitiveInputs(inputPlists)
              .addOutput(merged)
              .build(context);
        }
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
   * Creates action to merge multiple Info.plist files of a binary into a single Info.plist. The
   * merge action is necessary if there are more than one input plist files or we have a bundle ID
   * to stamp on the merged plist.
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
