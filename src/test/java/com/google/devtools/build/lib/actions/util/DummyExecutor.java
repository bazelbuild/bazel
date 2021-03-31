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
package com.google.devtools.build.lib.actions.util;

import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ActionExecutionContext.ShowSubcommands;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.clock.Clock;
import com.google.devtools.build.lib.testutil.ManualClock;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsProvider;
import javax.annotation.Nullable;

/** A dummy implementation of Executor. */
public class DummyExecutor implements Executor {

  private final FileSystem fileSystem;
  private final Path inputDir;
  private final ManualClock clock = new ManualClock();
  @Nullable private final OptionsProvider optionsProvider;

  public DummyExecutor(
      FileSystem fileSystem, Path inputDir, @Nullable OptionsProvider optionsProvider) {
    this.fileSystem = fileSystem;
    this.inputDir = inputDir;
    this.optionsProvider = optionsProvider;
  }

  public DummyExecutor(FileSystem fileSystem, Path inputDir) {
    this(fileSystem, inputDir, null);
  }

  public DummyExecutor() {
    this(/*fileSystem=*/ null, /*inputDir=*/ null);
  }

  @Override
  public FileSystem getFileSystem() {
    return fileSystem;
  }

  @Override
  public Path getExecRoot() {
    return inputDir;
  }

  @Override
  public Clock getClock() {
    return clock;
  }

  @Override
  public BugReporter getBugReporter() {
    return BugReporter.defaultInstance();
  }

  @Override
  public <T extends ActionContext> T getContext(Class<T> type) {
    throw new UnsupportedOperationException();
  }

  @Override
  public OptionsProvider getOptions() {
    if (optionsProvider != null) {
      return optionsProvider;
    }
    throw new UnsupportedOperationException();
  }

  @Override
  public ShowSubcommands reportsSubcommands() {
    throw new UnsupportedOperationException();
  }
}
