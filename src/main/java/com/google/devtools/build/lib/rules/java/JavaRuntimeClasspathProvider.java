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

package com.google.devtools.build.lib.rules.java;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import com.google.devtools.build.lib.syntax.SkylarkCallable;
import com.google.devtools.build.lib.syntax.SkylarkModule;

/**
 * Provider for the runtime classpath contributions of a Java binary.
 *
 * Used to exclude already-available artifacts from related binaries
 * (e.g. plugins).
 */
@Immutable
@SkylarkModule(name = "JavaRuntimeClasspathProvider", doc = "")
public final class JavaRuntimeClasspathProvider implements TransitiveInfoProvider {

  private final NestedSet<Artifact> runtimeClasspath;

  public JavaRuntimeClasspathProvider(NestedSet<Artifact> runtimeClasspath) {
    this.runtimeClasspath = runtimeClasspath;
  }

  /**
   * Returns the artifacts included on the runtime classpath of this binary.
   */
  @SkylarkCallable(name = "runtime_classpath", doc = "", structField = true)
  public NestedSet<Artifact> getRuntimeClasspath() {
    return runtimeClasspath;
  }
}
