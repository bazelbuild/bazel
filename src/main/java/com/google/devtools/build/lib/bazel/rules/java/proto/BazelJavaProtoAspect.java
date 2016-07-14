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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaSemantics;
import com.google.devtools.build.lib.rules.java.proto.JavaProtoAspect;

/** An Aspect which BazelJavaProtoLibrary injects to build Java SPEED protos. */
public class BazelJavaProtoAspect extends JavaProtoAspect {

  static final String SPEED_PROTO_RUNTIME_ATTR = "$aspect_java_lib";
  static final String SPEED_PROTO_RUNTIME_LABEL = "//third_party/protobuf:protobuf";

  public BazelJavaProtoAspect() {
    super(
        BazelJavaSemantics.INSTANCE,
        SPEED_PROTO_RUNTIME_ATTR,
        SPEED_PROTO_RUNTIME_LABEL,
        ImmutableList.<String>of(),
        null, /* jacocoAttr */
        ImmutableList.of("shared", "immutable"));
  }
}
