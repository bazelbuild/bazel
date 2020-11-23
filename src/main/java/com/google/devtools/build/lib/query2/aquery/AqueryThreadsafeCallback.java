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
package com.google.devtools.build.lib.query2.aquery;

import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.query2.NamedThreadSafeOutputFormatterCallback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.TargetAccessor;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import java.io.OutputStream;
import java.io.PrintStream;

/** Base class for aquery output callbacks. */
public abstract class AqueryThreadsafeCallback
    extends NamedThreadSafeOutputFormatterCallback<ConfiguredTargetValue> {
  protected final ExtendedEventHandler eventHandler;
  protected final AqueryOptions options;
  protected final PrintStream printStream;
  protected final SkyframeExecutor skyframeExecutor;
  protected final ConfiguredTargetValueAccessor accessor;

  AqueryThreadsafeCallback(
      ExtendedEventHandler eventHandler,
      AqueryOptions options,
      OutputStream out,
      SkyframeExecutor skyframeExecutor,
      TargetAccessor<ConfiguredTargetValue> accessor) {
    this.eventHandler = eventHandler;
    this.options = options;
    this.printStream = out == null ? null : new PrintStream(out);
    this.skyframeExecutor = skyframeExecutor;
    this.accessor = (ConfiguredTargetValueAccessor) accessor;
  }
}
