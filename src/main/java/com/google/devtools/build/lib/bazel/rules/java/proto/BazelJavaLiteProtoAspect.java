// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.rules.java.proto;

import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaSemantics;
import com.google.devtools.build.lib.rules.java.proto.JavaLiteProtoAspect;

/** An Aspect which BazelJavaLiteProtoLibrary injects to build Java Lite protos. */
public class BazelJavaLiteProtoAspect extends JavaLiteProtoAspect {

  public static final String DEFAULT_PROTO_TOOLCHAIN_LABEL =
      "@com_google_protobuf//:javalite_toolchain";

  public BazelJavaLiteProtoAspect(RuleDefinitionEnvironment env) {
    super(BazelJavaSemantics.INSTANCE, DEFAULT_PROTO_TOOLCHAIN_LABEL, env);
  }
}
