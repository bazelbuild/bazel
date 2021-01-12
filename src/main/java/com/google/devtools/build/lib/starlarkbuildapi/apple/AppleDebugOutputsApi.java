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

package com.google.devtools.build.lib.starlarkbuildapi.apple;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import com.google.devtools.build.lib.starlarkbuildapi.core.StructApi;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Dict;

/** A provider that holds debug outputs of an apple_binary target. */
@StarlarkBuiltin(
    name = "AppleDebugOutputs",
    category = DocCategory.PROVIDER,
    doc = "A provider that holds debug outputs of an apple_binary target.")
public interface AppleDebugOutputsApi<FileT extends FileApi> extends StructApi {

  @StarlarkMethod(
      name = "outputs_map",
      structField = true,
      doc =
          "A dictionary of: { arch: { output_type: file, output_type: file, ... } }, where 'arch'"
              + " is any Apple architecture such as 'arm64' or 'armv7', 'output_type' is a string"
              + " descriptor such as 'bitcode_symbols' or 'dsym_binary', and the file is the file"
              + " matching that descriptor for that architecture.")
  ImmutableMap<String, Dict<String, FileT>> getOutputsMap();
}
