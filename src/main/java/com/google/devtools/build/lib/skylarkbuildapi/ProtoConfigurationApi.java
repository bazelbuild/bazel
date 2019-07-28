// Copyright 2018 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * A configuration fragment representing protocol buffers.
 */
@SkylarkModule(
    name = "proto",
    category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT,
    doc = "A configuration fragment representing protocol buffers. "
              + "Do not use these fields, they will be removed after migrating "
              + "Protobuf rules to Starlark."
)
public interface ProtoConfigurationApi {
  @SkylarkCallable(
      name = "protoc_opts",
      doc = "Additional options to pass to the protobuf compiler. "
                + "Do not use this field, its only puprose is to help with migration of "
                + "Protobuf rules to Starlark.",
      structField = true)
  ImmutableList<String> protocOpts();

  @SkylarkCallable(
      name = "strict_deps",
      doc = "A string that specifies how to handle strict deps. Possible values: 'OFF', 'WARN', "
                + "'ERROR'. For more details see https://docs.bazel.build/versions/master/"
                + "command-line-reference.html#flag--strict_proto_deps"
                + "Do not use this field, its only puprose is to help with migration of "
                + "Protobuf rules to Starlark.",
      structField = true)
  String starlarkStrictDeps();
}
