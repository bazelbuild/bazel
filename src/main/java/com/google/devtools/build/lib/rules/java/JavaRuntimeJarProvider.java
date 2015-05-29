// Copyright 2015 Google Inc. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * This {@link com.google.devtools.build.lib.analysis.TransitiveInfoProvider} contains the .jar
 * files to be put on the runtime classpath by the configured target.
 *
 * <p>Unlike {@link com.google.devtools.build.lib.rules.java.JavaCompilationArgs#getRuntimeJars()},
 * it does not contain transitive runtime jars, only those produced by the configured target itself.
 *
 * <p>The reason why this class exists is that neverlink libraries do not contain the compiled jar
 * in {@link com.google.devtools.build.lib.rules.java.JavaCompilationArgs#getRuntimeJars()}.
 */
@Immutable
public final class JavaRuntimeJarProvider implements TransitiveInfoProvider {
  private final ImmutableList<Artifact> runtimeJars;

  public JavaRuntimeJarProvider(ImmutableList<Artifact> runtimeJars) {
    this.runtimeJars = runtimeJars;
  }

  public ImmutableList<Artifact> getRuntimeJars() {
    return runtimeJars;
  }
}
