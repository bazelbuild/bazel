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

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.SpawnActionContext;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Clock;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.OptionsClassProvider;

/**
 * A dummy implementation of Executor.
 */
public final class DummyExecutor implements Executor {
  private final Path inputDir;

  /**
   * @param inputDir
   */
  public DummyExecutor(Path inputDir) {
    this.inputDir = inputDir;
  }

  @Override
  public Path getExecRoot() {
    return inputDir;
  }

  @Override
  public Clock getClock() {
    throw new UnsupportedOperationException();
  }

  @Override
  public EventBus getEventBus() {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean getVerboseFailures() {
    throw new UnsupportedOperationException();
  }

  @Override
  public EventHandler getEventHandler() {
    throw new UnsupportedOperationException();
  }

  @Override
  public <T extends ActionContext> T getContext(Class<? extends T> type) {
    throw new UnsupportedOperationException();
  }

  @Override
  public SpawnActionContext getSpawnActionContext(String mnemonic) {
    throw new UnsupportedOperationException();
  }

  @Override
  public OptionsClassProvider getOptions() {
    throw new UnsupportedOperationException();
  }

  @Override
  public boolean reportsSubcommands() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void reportSubcommand(String reason, String message) {
    throw new UnsupportedOperationException();
  }
}