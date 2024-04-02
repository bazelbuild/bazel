// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.PackageContext;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue;
import com.google.devtools.build.lib.skyframe.config.NativeAndStarlarkFlags;
import com.google.devtools.build.lib.skyframe.config.ParsedFlagsValue;
import com.google.devtools.build.lib.skyframe.toolchains.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.state.StateMachine;
import com.google.devtools.common.options.OptionsParsingException;
import javax.annotation.Nullable;

/** Retrieves the parsed flags for a given platform. */
public class PlatformFlagsProducer implements StateMachine, PlatformInfoProducer.ResultSink {

  interface ResultSink {
    void acceptPlatformFlags(NativeAndStarlarkFlags flags);

    void acceptPlatformFlagsError(InvalidPlatformException error);

    void acceptPlatformFlagsError(OptionsParsingException error);
  }

  // -------------------- Input --------------------
  private final Label platformLabel;

  // -------------------- Output --------------------
  private final ResultSink sink;

  // -------------------- Sequencing --------------------
  private final StateMachine runAfter;

  // -------------------- Internal State --------------------
  @Nullable private RepositoryMapping platformRepositoryMapping;
  @Nullable private PlatformInfo platformInfo;
  @Nullable private NativeAndStarlarkFlags parsedFlags;

  PlatformFlagsProducer(Label platformLabel, ResultSink sink, StateMachine runAfter) {
    this.platformLabel = platformLabel;
    this.sink = sink;
    this.runAfter = runAfter;
  }

  @Override
  public StateMachine step(Tasks tasks) throws InterruptedException {
    // This needs a valid repository mapping in order to properly handle Starlark flags.
    tasks.lookUp(
        RepositoryMappingValue.key(this.platformLabel.getRepository()),
        this::acceptRepositoryMappingValue);
    // The platform info has the actual flags to process.
    return new PlatformInfoProducer(this.platformLabel, this, this::parsePlatformFlags);
  }

  private void acceptRepositoryMappingValue(SkyValue skyValue) {
    if (skyValue instanceof RepositoryMappingValue) {
      RepositoryMappingValue repositoryMappingValue = (RepositoryMappingValue) skyValue;
      this.platformRepositoryMapping = repositoryMappingValue.getRepositoryMapping();
      return;
    }

    throw new IllegalStateException("Unknown SkyValue type " + skyValue);
  }

  @Override
  public void acceptPlatformInfo(PlatformInfo info) {
    this.platformInfo = info;
  }

  @Override
  public void acceptPlatformInfoError(InvalidPlatformException error) {
    sink.acceptPlatformFlagsError(error);
  }

  private StateMachine parsePlatformFlags(Tasks tasks) {
    if (this.platformRepositoryMapping == null || this.platformInfo == null) {
      return DONE; // There was an error.
    }

    ImmutableList<String> flags = this.platformInfo.flags();
    if (flags.isEmpty()) {
      // Nothing to do, so skip out now.
      return this.runAfter;
    }

    PackageContext packageContext =
        PackageContext.of(platformLabel.getPackageIdentifier(), this.platformRepositoryMapping);
    ParsedFlagsValue.Key parsedFlagsKey = ParsedFlagsValue.Key.create(flags, packageContext);
    tasks.lookUp(parsedFlagsKey, OptionsParsingException.class, this::acceptParsedFlagsValue);
    return this::handleParsedFlags;
  }

  private void acceptParsedFlagsValue(
      @Nullable SkyValue value, @Nullable OptionsParsingException exception) {
    if (value != null && value instanceof ParsedFlagsValue) {
      this.parsedFlags = ((ParsedFlagsValue) value).flags();
      return;
    }
    if (exception != null) {
      sink.acceptPlatformFlagsError(exception);
      return;
    }
    throw new IllegalStateException("Both value and exception are null");
  }

  private StateMachine handleParsedFlags(Tasks tasks) {
    if (this.parsedFlags == null) {
      return DONE; // There was an error.
    }
    sink.acceptPlatformFlags(this.parsedFlags);
    return this.runAfter;
  }
}
