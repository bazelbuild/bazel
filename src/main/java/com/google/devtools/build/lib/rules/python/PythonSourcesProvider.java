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
package com.google.devtools.build.lib.rules.python;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.syntax.SkylarkModule;

/**
 * A provider interface for configured targets that provide source files to
 * Python targets.
 */
@Immutable
@SkylarkModule(name = "PythonSourcesProvider", doc = "")
public final class PythonSourcesProvider implements TransitiveInfoProvider {
  /** The name of the field in Skylark used to access this class. */
  public static final String SKYLARK_NAME = "py";

  private final NestedSet<Artifact> transitivePythonSources;
  private final boolean usesSharedLibraries;

  public PythonSourcesProvider(NestedSet<Artifact> transitivePythonSources,
      boolean usesSharedLibraries) {
    this.transitivePythonSources = transitivePythonSources;
    this.usesSharedLibraries = usesSharedLibraries;
  }

  /**
   * Returns the Python sources in the transitive closure of this target.
   */
  @SkylarkCallable(
      name = "transitive_sources", doc = "The transitive set of Python sources", structField = true)
  public NestedSet<Artifact> getTransitivePythonSources() {
    return transitivePythonSources;
  }

  /**
   * Returns true if this target transitively depends on any shared libraries.
   */
  public boolean usesSharedLibraries() {
    return usesSharedLibraries;
  }
}
