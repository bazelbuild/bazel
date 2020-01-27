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

package com.google.devtools.build.lib.skylarkbuildapi.apple;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkbuildapi.core.StructApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * A provider that holds debug outputs of an apple_binary target.
 */
@SkylarkModule(
    name = "AppleDebugOutputs",
    category = SkylarkModuleCategory.PROVIDER,
    doc = "A provider that holds debug outputs of an apple_binary target."
)
public interface AppleDebugOutputsApi<FileT extends FileApi> extends StructApi {

  @SkylarkCallable(
      name = "outputs_map",
      structField = true,
      doc =
          "A dictionary of: { arch: { output_type: file, output_type: file, ... } }, where 'arch'"
              + " is any Apple architecture such as 'arm64' or 'armv7', 'output_type' is a string"
              + " descriptor such as 'bitcode_symbols' or 'dsym_binary', and the file is the file"
              + " matching that descriptor for that architecture.")
  ImmutableMap<String, ImmutableMap<String, FileT>> getOutputsMap();
}
