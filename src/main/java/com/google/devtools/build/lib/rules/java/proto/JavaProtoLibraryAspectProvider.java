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
import com.google.devtools.build.lib.analysis.TransitiveInfoProvider;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.rules.java.JavaCompilationArgsProvider;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;

/** A provider used to communicate information between java_proto_library and its aspect. */
@AutoCodec
public class JavaProtoLibraryAspectProvider implements TransitiveInfoProvider {
  private final NestedSet<Artifact> jars;

  /**
   * List of jars to be used as "direct" when java_xxx_proto_library.strict_deps = 0.
   *
   * <p>Contains all transitively generated protos and all proto runtimes (but not the runtime's own
   * dependencies).
   */
  private final JavaCompilationArgsProvider nonStrictCompArgs;

  public JavaProtoLibraryAspectProvider(
      NestedSet<Artifact> jars, JavaCompilationArgsProvider nonStrictCompArgs) {
    this.jars = jars;
    this.nonStrictCompArgs = nonStrictCompArgs;
  }

  public NestedSet<Artifact> getJars() {
    return jars;
  }

  public JavaCompilationArgsProvider getNonStrictCompArgs() {
    return nonStrictCompArgs;
  }
}
