// Copyright 2017 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec.util;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata.MiddlemanType;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.RunfilesSupplier;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.packages.AspectDescriptor;
import javax.annotation.Nullable;

/**
 * Fake implementation of {@link ActionExecutionMetadata} for testing.
 */
public final class FakeOwner implements ActionExecutionMetadata {
  private final String mnemonic;
  private final String progressMessage;
  @Nullable private final String ownerLabel;
  @Nullable private final PlatformInfo platform;
  ImmutableMap<String, String> execProperties;

  public FakeOwner(
      String mnemonic,
      String progressMessage,
      @Nullable String ownerLabel,
      @Nullable PlatformInfo platform,
      ImmutableMap<String, String> execProperties) {
    this.mnemonic = mnemonic;
    this.progressMessage = progressMessage;
    this.ownerLabel = ownerLabel;
    this.platform = platform;
    this.execProperties = execProperties;
  }

  public FakeOwner(
      String mnemonic,
      String progressMessage,
      @Nullable String ownerLabel,
      @Nullable PlatformInfo platform) {
    this(mnemonic, progressMessage, ownerLabel, platform, ImmutableMap.of());
  }

  public FakeOwner(String mnemonic, String progressMessage, @Nullable String ownerLabel) {
    this(mnemonic, progressMessage, ownerLabel, null);
  }

  public FakeOwner(String mnemonic, String progressMessage) {
    this(mnemonic, progressMessage, null);
  }

  @Override
  public ActionOwner getOwner() {
    return ActionOwner.create(
        ownerLabel == null ? null : Label.parseAbsoluteUnchecked(ownerLabel),
        /*aspectDescriptors=*/ ImmutableList.<AspectDescriptor>of(),
        /*location=*/ null,
        mnemonic,
        /*targetKind=*/ null,
        "configurationChecksum",
        /* configuration=*/ null,
        "additionalProgressInfo",
        /* execProperties=*/ ImmutableMap.of(),
        null);
  }

  @Override
  public boolean isShareable() {
    return false;
  }

  @Override
  public String getMnemonic() {
    return mnemonic;
  }

  @Override
  public String getProgressMessage() {
    return progressMessage;
  }

  @Override
  public boolean inputsDiscovered() {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean discoversInputs() {
    throw new UnsupportedOperationException();
  }

  @Override
  public NestedSet<Artifact> getTools() {
    throw new UnsupportedOperationException();
  }

  @Override
  public NestedSet<Artifact> getInputs() {
    throw new UnsupportedOperationException();
  }

  @Override
  public RunfilesSupplier getRunfilesSupplier() {
    throw new UnsupportedOperationException();
  }

  @Override
  public ImmutableSet<Artifact> getOutputs() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Iterable<String> getClientEnvironmentVariables() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Artifact getPrimaryInput() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Artifact getPrimaryOutput() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Iterable<Artifact> getMandatoryInputs() {
    throw new UnsupportedOperationException();
  }

  @Override
  public String getKey(ActionKeyContext actionKeyContext) {
    return "MockOwner.getKey";
  }

  @Override
  public String describeKey() {
    throw new UnsupportedOperationException();
  }

  @Override
  public String prettyPrint() {
    throw new UnsupportedOperationException();
  }

  @Override
  public String describe() {
    return getProgressMessage();
  }

  @Override
  public Iterable<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext) {
    return ImmutableList.of();
  }

  @Override
  public ImmutableSet<Artifact> getMandatoryOutputs() {
    throw new UnsupportedOperationException();
  }

  @Override
  public MiddlemanType getActionType() {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean shouldReportPathPrefixConflict(ActionAnalysisMetadata action) {
    throw new UnsupportedOperationException();
  }

  @Override
  public ImmutableMap<String, String> getExecProperties() {
    return execProperties;
  }

  @Nullable
  @Override
  public PlatformInfo getExecutionPlatform() {
    return platform;
  }
}
