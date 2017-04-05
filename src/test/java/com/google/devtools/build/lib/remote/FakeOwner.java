// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.remote;

import static org.mockito.Mockito.mock;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionExecutionMetadata;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.RunfilesSupplier;

/** A fake implementation of ActionExecutionMetadata as needed for SpawnRunner test. */
final class FakeOwner implements ActionExecutionMetadata {
  private final String mnemonic;
  private final String progressMessage;

  FakeOwner(String mnemonic, String progressMessage) {
    this.mnemonic = mnemonic;
    this.progressMessage = progressMessage;
  }

  @Override
  public ActionOwner getOwner() {
    return mock(ActionOwner.class);
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
  public Iterable<Artifact> getTools() {
    throw new UnsupportedOperationException();
  }

  @Override
  public Iterable<Artifact> getInputs() {
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
  public String getKey() {
    throw new UnsupportedOperationException();
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
  public Iterable<Artifact> getInputFilesForExtraAction(
      ActionExecutionContext actionExecutionContext) {
    return ImmutableList.<Artifact>of();
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
}