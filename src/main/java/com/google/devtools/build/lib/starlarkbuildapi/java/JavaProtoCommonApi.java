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

package com.google.devtools.build.lib.starlarkbuildapi.java;

import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.StarlarkRuleContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.TransitiveInfoCollectionApi;
import com.google.devtools.build.lib.starlarkbuildapi.platform.ConstraintValueInfoApi;
import net.starlark.java.annot.Param;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.StarlarkValue;

/** Helper class for Java proto compilation. */
@StarlarkBuiltin(name = "java_proto_common", doc = "Helper class for Java proto compilation.")
public interface JavaProtoCommonApi<
        FileT extends FileApi,
        ConstraintValueT extends ConstraintValueInfoApi,
        StarlarkRuleContextT extends StarlarkRuleContextApi<ConstraintValueT>,
        TransitiveInfoCollectionT extends TransitiveInfoCollectionApi>
    extends StarlarkValue {

  @StarlarkMethod(
      name = "create_java_lite_proto_compile_action",
      // This function is experimental for now.
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = true, named = false, doc = "The rule context."),
        @Param(name = "target", positional = true, named = false, doc = "The target."),
        @Param(name = "src_jar", positional = false, named = true),
        @Param(name = "proto_toolchain_attr", positional = false, named = true),
        @Param(name = "flavour", positional = false, named = true, defaultValue = "'java'"),
      })
  void createProtoCompileAction(
      StarlarkRuleContextT starlarkRuleContext,
      TransitiveInfoCollectionT target,
      FileT sourceJar,
      String protoToolchainAttr,
      String flavour)
      throws EvalException;

  @StarlarkMethod(
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
            doc = "The proto_library target."),
      })
  boolean hasProtoSources(TransitiveInfoCollectionT target);

  @StarlarkMethod(
      name = "toolchain_deps",
      // This function is experimental for now.
      documented = false,
      parameters = {
        @Param(name = "ctx", positional = true, named = false, doc = "The rule context."),
        @Param(name = "proto_toolchain_attr", positional = false, named = true)
      })
  JavaInfoApi<FileT, ?> getRuntimeToolchainProvider(
      StarlarkRuleContextT starlarkRuleContext, String protoToolchainAttr) throws EvalException;
}
