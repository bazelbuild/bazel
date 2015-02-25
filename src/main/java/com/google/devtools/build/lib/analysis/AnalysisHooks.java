// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.pkgcache.PackageManager;

/**
 * This interface resolves target - configuration pairs to {@link ConfiguredTarget} instances.
 *
 * <p>This interface is used to provide analysis phase functionality to actions that need it in
 * the execution phase.
 */
public interface AnalysisHooks {
  /**
   * Returns the package manager used during the analysis phase.
   */
  PackageManager getPackageManager();

  /**
   * Resolves an existing configured target. Returns null if it is not in the cache.
   */
  ConfiguredTarget getExistingConfiguredTarget(Target target, BuildConfiguration configuration);
}
