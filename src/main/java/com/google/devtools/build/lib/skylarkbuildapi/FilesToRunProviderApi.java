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

import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;
import com.google.devtools.build.lib.syntax.StarlarkValue;
import javax.annotation.Nullable;

/** Returns information about executables produced by a target and the files needed to run it. */
@SkylarkModule(name = "FilesToRunProvider", doc = "", category = SkylarkModuleCategory.PROVIDER)
public interface FilesToRunProviderApi<FileT extends FileApi> extends StarlarkValue {

  @SkylarkCallable(
      name = "executable",
      doc = "The main executable or None if it does not exist.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  FileT getExecutable();

  @SkylarkCallable(
      name = "runfiles_manifest",
      doc = "The runfiles manifest or None if it does not exist.",
      structField = true,
      allowReturnNones = true)
  @Nullable
  FileT getRunfilesManifest();
}
