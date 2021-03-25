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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.StarlarkValue;

/** Info object about outputs of a Java rule. */
@StarlarkBuiltin(
    name = "java_output_jars",
    category = DocCategory.PROVIDER,
    doc = "Information about outputs of a Java rule.")
public interface JavaRuleOutputJarsProviderApi<OutputJarT extends OutputJarApi<?>>
    extends StarlarkValue {

  @StarlarkMethod(name = "jars", doc = "A list of jars the rule outputs.", structField = true)
  ImmutableList<OutputJarT> getOutputJars();

  @StarlarkMethod(
      name = "jdeps",
      doc =
          "A manifest proto file. The protobuf file containing the manifest generated from "
              + "JavaBuilder. This function returns a value when exactly one manifest proto file is"
              + " present in the outputs.  Deprecated: Use jars[i].jdeps.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  @Deprecated
  FileApi getJdeps();

  @StarlarkMethod(
      name = "native_headers",
      doc =
          "A jar containing CC header files supporting native method implementation. "
              + " This function returns a value when exactly one native headers jar file is"
              + " present in the outputs. Deprecated: Use jars[i].native_headers_jar.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  @Deprecated
  FileApi getNativeHeaders();
}
