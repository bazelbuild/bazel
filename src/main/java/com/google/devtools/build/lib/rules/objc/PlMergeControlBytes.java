// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteSource;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos;
import com.google.devtools.build.xcode.plmerge.proto.PlMergeProtos.Control;
import java.io.IOException;
import java.io.InputStream;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * A byte source that can be used the generate a control file for the tool plmerge.
 */
public final class PlMergeControlBytes extends ByteSource {

  private final NestedSet<Artifact> inputPlists;
  private final NestedSet<Artifact> immutableInputPlists;
  @Nullable private final String primaryBundleId;
  @Nullable private final String fallbackBundleId;
  @Nullable private final Map<String, String> variableSubstitutions;
  @Nullable private final String executableName;
  private final Artifact mergedPlist;
  private final OutputFormat outputFormat;

  /**
   * Creates a control based on the passed bundling's plists and values.
   *
   * @param bundling bundle for which to create a merged plist
   * @param mergedPlist the plist that should be created as an output
   */
  static PlMergeControlBytes fromBundling(Bundling bundling, Artifact mergedPlist) {

    NestedSetBuilder<Artifact> immutableInputPlists = NestedSetBuilder.stableOrder();
    if (bundling.getAutomaticInfoPlist() != null) {
      immutableInputPlists.add(bundling.getAutomaticInfoPlist());
    }

    return new PlMergeControlBytes(
        NestedSetBuilder.<Artifact>stableOrder()
            .addTransitive(bundling.getBundleInfoplistInputs())
            .build(),
        immutableInputPlists.build(),
        bundling.getPrimaryBundleId(),
        bundling.getFallbackBundleId(),
        bundling.variableSubstitutions(),
        bundling.getExecutableName(),
        mergedPlist,
        OutputFormat.BINARY);
  }

  /**
   * Creates a control that merges the given plists into the merged plist.
   */
  static PlMergeControlBytes fromPlists(
      NestedSet<Artifact> inputPlists,
      Artifact mergedPlist,
      OutputFormat outputFormat) {
    return new PlMergeControlBytes(
        inputPlists,
        NestedSetBuilder.<Artifact>emptySet(Order.STABLE_ORDER),
        null,
        null,
        ImmutableMap.<String, String>of(),
        null,
        mergedPlist,
        outputFormat);
  }

  private PlMergeControlBytes(
      NestedSet<Artifact> inputPlists,
      NestedSet<Artifact> immutableInputPlists,
      @Nullable String primaryBundleId,
      @Nullable String fallbackBundleId,
      @Nullable Map<String, String> variableSubstitutions,
      @Nullable String executableName,
      Artifact mergedPlist,
      OutputFormat outputFormat) {
    this.inputPlists = inputPlists;
    this.immutableInputPlists = immutableInputPlists;
    this.primaryBundleId = primaryBundleId;
    this.fallbackBundleId = fallbackBundleId;
    this.variableSubstitutions = variableSubstitutions;
    this.executableName = executableName;
    this.mergedPlist = mergedPlist;
    this.outputFormat = Preconditions.checkNotNull(outputFormat);
  }

  @Override
  public InputStream openStream() throws IOException {
    return control().toByteString().newInput();
  }

  private Control control() {
    PlMergeProtos.Control.Builder control =
        PlMergeProtos.Control.newBuilder()
            .addAllSourceFile(Artifact.toExecPaths(inputPlists))
            .addAllImmutableSourceFile(Artifact.toExecPaths(immutableInputPlists))
            .putAllVariableSubstitutionMap(variableSubstitutions)
            .setOutFile(mergedPlist.getExecPathString());

    if (primaryBundleId != null) {
      control.setPrimaryBundleId(primaryBundleId);
    }

    if (fallbackBundleId != null) {
      control.setFallbackBundleId(fallbackBundleId);
    }

    if (executableName != null) {
      control.setExecutableName(executableName);
    }

    control.setOutputFormat(outputFormat.getProtoOutputFormat());

    return control.build();
  }

  /**
   * Plist output formats.
   */
  public enum OutputFormat {
    BINARY(Control.OutputFormat.BINARY),
    XML(Control.OutputFormat.XML);

    private final Control.OutputFormat protoOutputFormat;

    private OutputFormat(Control.OutputFormat protoOutputFormat) {
      this.protoOutputFormat = protoOutputFormat;
    }

    Control.OutputFormat getProtoOutputFormat() {
      return protoOutputFormat;
    }
  }
}
