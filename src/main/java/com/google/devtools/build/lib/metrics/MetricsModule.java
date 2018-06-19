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
package com.google.devtools.build.lib.metrics;

import com.google.devtools.build.lib.runtime.BlazeModule;
import com.google.devtools.build.lib.runtime.CommandEnvironment;

/**
 * A blaze module that installs metrics instrumentations and issues a {@link BuildMetricsEvent} at
 * the end of the build.
 */
public class MetricsModule extends BlazeModule {

  @Override
  public void beforeCommand(CommandEnvironment env) {
    MetricsCollector.installInEnv(env);
  }

  @Override
  public void afterCommand() {}
}
