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

package com.google.devtools.build.lib.bazel.repository.skylark;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;

import java.util.Map;

/**
 * A Skylark structure to deliver information about the system we are running on.
 */
@Immutable
@SkylarkModule(
  name = "repository_os",
  doc = "Various data about the current platform Bazel is running on."
)
final class SkylarkOS {

  private final ImmutableMap<String, String> environ;

  SkylarkOS(Map<String, String> environ) {
    this.environ = ImmutableMap.copyOf(environ);
  }

  @SkylarkCallable(name = "environ", structField = true, doc = "The list of environment variables.")
  public ImmutableMap<String, String> getEnviron() {
    return environ;
  }

  @SkylarkCallable(
    name = "name",
    structField = true,
    doc = "A string identifying the current system Bazel is running on."
  )
  public String getName() {
    return System.getProperty("os.name").toLowerCase();
  }
}
