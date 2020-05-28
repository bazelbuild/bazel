// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.starlark;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkbuildapi.repository.StarlarkOSApi;
import java.util.Map;

/** A Starlark structure to deliver information about the system we are running on. */
@Immutable
final class StarlarkOS implements StarlarkOSApi {

  private final ImmutableMap<String, String> environ;

  StarlarkOS(Map<String, String> environ) {
    this.environ = ImmutableMap.copyOf(environ);
  }

  @Override
  public boolean isImmutable() {
    return true; // immutable and Starlark-hashable
  }

  @Override
  public ImmutableMap<String, String> getEnvironmentVariables() {
    return environ;
  }

  @Override
  public String getName() {
    return System.getProperty("os.name").toLowerCase();
  }
}
