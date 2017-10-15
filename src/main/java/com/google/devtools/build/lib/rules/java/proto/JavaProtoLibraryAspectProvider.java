// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.java.proto;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.TransitiveInfoProviderMap;
import com.google.devtools.build.lib.analysis.WrappingProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgs;

/** A provider used to communicate information between java_proto_library and its aspect. */
public class JavaProtoLibraryAspectProvider implements WrappingProvider {
  private final TransitiveInfoProviderMap transitiveInfoProviderMap;
  private final NestedSet<Artifact> jars;

  /**
   * List of jars to be used as "direct" when java_xxx_proto_library.strict_deps = 0.
   *
   * <p>Contains all transitively generated protos and all proto runtimes (but not the runtime's own
   * dependencies).
   */
  private final JavaCompilationArgs nonStrictCompArgs;

  public JavaProtoLibraryAspectProvider(
      TransitiveInfoProviderMap transitiveInfoProviderMap,
      NestedSet<Artifact> jars,
      JavaCompilationArgs nonStrictCompArgs) {
    this.transitiveInfoProviderMap = transitiveInfoProviderMap;
    this.jars = jars;
    this.nonStrictCompArgs = nonStrictCompArgs;
  }

  @Override
  public TransitiveInfoProviderMap getTransitiveInfoProviderMap() {
    return transitiveInfoProviderMap;
  }

  public NestedSet<Artifact> getJars() {
    return jars;
  }

  public JavaCompilationArgs getNonStrictCompArgs() {
    return nonStrictCompArgs;
  }
}
