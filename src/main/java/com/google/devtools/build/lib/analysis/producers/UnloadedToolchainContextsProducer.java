// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.producers;

import static com.google.devtools.build.lib.packages.ExecGroup.DEFAULT_EXEC_GROUP_NAME;

import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.packages.ExecGroup;
import com.google.devtools.build.lib.skyframe.toolchains.NoMatchingPlatformException;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainContextKey;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainException;
import com.google.devtools.build.lib.skyframe.toolchains.UnloadedToolchainContext;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import java.util.Map;
import javax.annotation.Nullable;

final class UnloadedToolchainContextsProducer implements StateMachine {
  interface ResultSink {
    void acceptUnloadedToolchainContexts(
        @Nullable ToolchainCollection<UnloadedToolchainContext> unloadedToolchainContexts);

    void acceptUnloadedToolchainContextsError(ToolchainException error);
  }

  // -------------------- Input --------------------
  private final UnloadedToolchainContextsInputs unloadedToolchainContextsInputs;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Sequencing --------------------
  private final StateMachine runAfter;

  // -------------------- Internal State --------------------
  private ToolchainCollection.Builder<UnloadedToolchainContext> toolchainContextsBuilder;
  private boolean toolchainContextsHasError = false;

  UnloadedToolchainContextsProducer(
      UnloadedToolchainContextsInputs unloadedToolchainContextsInputs,
      ResultSink sink,
      StateMachine runAfter) {
    this.unloadedToolchainContextsInputs = unloadedToolchainContextsInputs;
    this.sink = sink;
    this.runAfter = runAfter;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    var defaultToolchainContextKey = unloadedToolchainContextsInputs.targetToolchainContextKey();
    if (defaultToolchainContextKey == null) {
      // Doesn't use toolchain resolution and short-circuits.
      sink.acceptUnloadedToolchainContexts(null);
      return runAfter;
    }

    this.toolchainContextsBuilder =
        ToolchainCollection.builderWithExpectedSize(
            unloadedToolchainContextsInputs.execGroups().size() + 1);

    tasks.lookUp(
        defaultToolchainContextKey,
        ToolchainException.class,
        new ToolchainContextLookupCallback(DEFAULT_EXEC_GROUP_NAME));

    var keyBuilder =
        ToolchainContextKey.key()
            .configurationKey(defaultToolchainContextKey.configurationKey())
            .debugTarget(defaultToolchainContextKey.debugTarget());

    for (Map.Entry<String, ExecGroup> entry :
        unloadedToolchainContextsInputs.execGroups().entrySet()) {
      var execGroup = entry.getValue();
      tasks.lookUp(
          keyBuilder
              .toolchainTypes(execGroup.toolchainTypes())
              .execConstraintLabels(execGroup.execCompatibleWith())
              .build(),
          ToolchainException.class,
          new ToolchainContextLookupCallback(entry.getKey()));
    }

    return this::buildToolchainContexts;
  }

  private class ToolchainContextLookupCallback
      implements StateMachine.ValueOrExceptionSink<ToolchainException> {
    private final String execGroupName;

    private ToolchainContextLookupCallback(String execGroupName) {
      this.execGroupName = execGroupName;
    }

    @Override
    public void acceptValueOrException(
        @Nullable SkyValue value, @Nullable ToolchainException error) {
      if (value != null) {
        var unloadedToolchainContext = (UnloadedToolchainContext) value;
        var errorData = unloadedToolchainContext.errorData();
        if (errorData != null) {
          handleError(new NoMatchingPlatformException(errorData));
          return;
        }
        toolchainContextsBuilder.addContext(execGroupName, unloadedToolchainContext);
        return;
      }
      if (error != null) {
        handleError(error);
        return;
      }
      throw new IllegalArgumentException("both inputs were null");
    }
  }

  private void handleError(ToolchainException error) {
    if (!toolchainContextsHasError) { // Only propagates the first error.
      toolchainContextsHasError = true;
      sink.acceptUnloadedToolchainContextsError(error);
    }
  }

  private StateMachine buildToolchainContexts(Tasks tasks) {
    if (toolchainContextsHasError) {
      return runAfter;
    }
    sink.acceptUnloadedToolchainContexts(toolchainContextsBuilder.build());
    return runAfter;
  }
}
