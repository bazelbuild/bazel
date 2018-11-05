// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2;


import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.query2.output.CqueryOptions;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Parents class for cquery output callbacks. Handles names and outputting contents of result list
 * that is populated by child classes.
 */
public abstract class CqueryThreadsafeCallback
    extends NamedThreadSafeOutputFormatterCallback<ConfiguredTarget> {

  protected final ExtendedEventHandler eventHandler;
  protected final CqueryOptions options;
  protected PrintStream printStream = null;
  protected final SkyframeExecutor skyframeExecutor;
  protected final ConfiguredTargetAccessor accessor;

  private final List<String> result = new ArrayList<>();

  CqueryThreadsafeCallback(
      ExtendedEventHandler eventHandler,
      CqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTarget> accessor) {
    this.eventHandler = eventHandler;
    this.options = options;
    if (out != null) {
      this.printStream = new PrintStream(out);
    }
    this.skyframeExecutor = skyframeExecutor;
    this.accessor = (ConfiguredTargetAccessor) accessor;
  }

  public void addResult(String string) {
    result.add(string);
  }

  @VisibleForTesting
  public List<String> getResult() {
    return result;
  }

  @Override
  public void close(boolean failFast) throws InterruptedException, IOException {
    if (!failFast && printStream != null) {
      result.forEach(printStream::println);
    }
  }
}

