// Copyright 2014 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

import javax.annotation.Nullable;

/**
 * Provides information about jar files produced by a Java rule.
 */
@Immutable
public final class JavaRuleOutputJarsProvider implements TransitiveInfoProvider {
  @Nullable private final Artifact classJar;
  @Nullable private final Artifact iJar;
  @Nullable private final Artifact srcJar;

  public JavaRuleOutputJarsProvider(
      @Nullable Artifact classJar, @Nullable Artifact iJar, @Nullable Artifact srcJar) {
    this.classJar = classJar;
    this.iJar = iJar;
    this.srcJar = srcJar;
  }

  @Nullable
  public Artifact getClassJar() {
    return classJar;
  }

  @Nullable
  public Artifact getIJar() {
    return iJar;
  }

  @Nullable
  public Artifact getSrcJar() {
    return srcJar;
  }
}
