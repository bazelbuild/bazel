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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;

/**
 * A Provider describing the java sources directly belonging to a java rule.
 */
@Immutable
public final class JavaSourceInfoProvider implements TransitiveInfoProvider {

  private final NestedSet<Artifact> sources;

  public JavaSourceInfoProvider(NestedSet<Artifact> sources) {
    Preconditions.checkNotNull(sources);
    this.sources = sources;
  }

  /**
   * Gets the original Java source artifacts, which may be .java, source .jar, or .srcjar files.
   * The .jars and .srcjars should contain java sources, but may include other files also.
   * 
   * @return the source artifacts for this JavaSourceInfoProvider
   */
  public NestedSet<Artifact> getSources() {
    return sources;
  }
}
