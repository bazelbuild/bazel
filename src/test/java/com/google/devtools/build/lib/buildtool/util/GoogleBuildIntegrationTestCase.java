// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool.util;

import com.google.devtools.build.lib.bazel.BazelRepositoryModule;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.NoSpawnCacheModule;

/**
 * Base class for some integration tests.
 *
 * <p>This class sets up Bazel-specific modules in the Bazel source tree, and Google-specific
 * modules in the internal source tree. Having "Google" in the class name means nothing for Bazel,
 * but renaming this class would require some work (which may or may not be much).
 *
 * <p>Having this class in the Bazel source tree enables maintaining the same class hierarchy
 * inside and outside of Google, and enables importing Bazel changes into Google's source tree.
 */
public abstract class GoogleBuildIntegrationTestCase extends BuildIntegrationTestCase {
  @Override
  protected BlazeRuntime.Builder getRuntimeBuilder() throws Exception {
    BlazeRuntime.Builder builder = super.getRuntimeBuilder();
    builder
        .addBlazeModule(new BazelRepositoryModule())
        .addBlazeModule(new NoSpawnCacheModule());
    return builder;
  }
}
