// Copyright 2015 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.analysis.skylark.SkylarkApiProvider;
import com.google.devtools.build.lib.packages.SkylarkProviderIdentifier;
import com.google.devtools.build.lib.skylarkbuildapi.java.JavaSkylarkApiProviderApi;

/**
 * A class that exposes the Java providers to Skylark. It is intended to provide a simple and stable
 * interface for Skylark users.
 */
public final class JavaSkylarkApiProvider extends SkylarkApiProvider
    implements JavaSkylarkApiProviderApi<Artifact> {
  /** The name of the field in Skylark used to access this class. */
  public static final String NAME = "java";
  /** The name of the field in Skylark proto aspects used to access this class. */
  public static final SkylarkProviderIdentifier SKYLARK_NAME =
      SkylarkProviderIdentifier.forLegacy(NAME);

  /**
   * Creates a Skylark API provider that reads information from its associated target's providers.
   */
  public static JavaSkylarkApiProvider fromRuleContext() {
    return new JavaSkylarkApiProvider();
  }
}
