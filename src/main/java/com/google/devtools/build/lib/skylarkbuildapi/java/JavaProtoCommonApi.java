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

package com.google.devtools.build.lib.skylarkbuildapi.java;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.SkylarkRuleContextApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.TransitiveInfoCollectionApi;
import com.google.devtools.build.lib.skylarkinterface.Param;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.StarlarkValue;

/** Helper class for Java proto compilation. */
@SkylarkModule(name = "java_proto_common", doc = "Helper class for Java proto compilation.")
public interface JavaProtoCommonApi<
        FileT extends FileApi,
        SkylarkRuleContextT extends SkylarkRuleContextApi,
        TransitiveInfoCollectionT extends TransitiveInfoCollectionApi>
    extends StarlarkValue {

  @SkylarkCallable(
      name = "create_java_lite_proto_compile_action",
      // This function is experimental for now.
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(
            name = "target",
            positional = true,
            named = false,
            type = TransitiveInfoCollectionApi.class,
            doc = "The target."),
        @Param(name = "src_jar", positional = false, named = true, type = FileApi.class),
        @Param(
            name = "proto_toolchain_attr",
            positional = false,
            named = true,
            type = String.class),
        @Param(
            name = "flavour",
            positional = false,
            named = true,
            type = String.class,
            defaultValue = "'java'")
      })
  void createProtoCompileAction(
      SkylarkRuleContextT skylarkRuleContext,
      TransitiveInfoCollectionT target,
      FileT sourceJar,
      String protoToolchainAttr,
      String flavour)
      throws EvalException;

  @SkylarkCallable(
      name = "has_proto_sources",
      doc =
          "Returns whether the given proto_library target contains proto sources. If there are no"
              + " sources it means that the proto_library is an alias library, which exports its"
              + " dependencies.",
      parameters = {
        @Param(
            name = "target",
            positional = true,
            named = false,
            type = TransitiveInfoCollectionApi.class,
            doc = "The proto_library target."),
      })
  boolean hasProtoSources(TransitiveInfoCollectionT target);

  @SkylarkCallable(
      name = "toolchain_deps",
      // This function is experimental for now.
      documented = false,
      parameters = {
        @Param(
            name = "ctx",
            positional = true,
            named = false,
            type = SkylarkRuleContextApi.class,
            doc = "The rule context."),
        @Param(name = "proto_toolchain_attr", positional = false, named = true, type = String.class)
      })
  JavaInfoApi<FileT> getRuntimeToolchainProvider(
      SkylarkRuleContextT skylarkRuleContext, String protoToolchainAttr) throws EvalException;
}
