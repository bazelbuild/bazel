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

import com.google.devtools.build.lib.analysis.platform.PlatformValue;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.build.skyframe.state.StateMachine.ValueOrException2Sink;
import com.google.devtools.common.options.OptionsParsingException;
import javax.annotation.Nullable;

/** Retrieves {@link PlatformValue} for a given platform. */
final class PlatformProducer
    implements StateMachine,
        ValueOrException2Sink<InvalidPlatformException, OptionsParsingException> {

  interface ResultSink {
    void acceptPlatformValue(PlatformValue value);

    void acceptPlatformInfoError(InvalidPlatformException error);

    void acceptOptionsParsingError(OptionsParsingException error);
  }

  // -------------------- Input --------------------
  private final Label platformLabel;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Sequencing --------------------
  private final StateMachine runAfter;

  PlatformProducer(Label platformLabel, ResultSink sink, StateMachine runAfter) {
    this.platformLabel = platformLabel;
    this.sink = sink;
    this.runAfter = runAfter;
  }

  @Override
  public StateMachine step(Tasks tasks) {
    tasks.lookUp(
        PlatformValue.key(platformLabel),
        InvalidPlatformException.class,
        OptionsParsingException.class,
        this);
    return runAfter;
  }

  @Override
  public void acceptValueOrException2(
      @Nullable SkyValue value,
      @Nullable InvalidPlatformException invalidPlatformException,
      @Nullable OptionsParsingException optionsParsingException) {
    if (value != null) {
      sink.acceptPlatformValue((PlatformValue) value);
    } else if (invalidPlatformException != null) {
      sink.acceptPlatformInfoError(invalidPlatformException);
    } else {
      sink.acceptOptionsParsingError(optionsParsingException);
    }
  }
}
