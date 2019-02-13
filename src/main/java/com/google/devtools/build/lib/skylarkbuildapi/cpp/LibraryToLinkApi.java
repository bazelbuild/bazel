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

package com.google.devtools.build.lib.skylarkbuildapi.cpp;

import com.google.devtools.build.lib.skylarkbuildapi.FileApi;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * A library the user can link to. This is different from a simple linker input in that it also has
 * a library identifier.
 */
@SkylarkModule(
    name = "LibraryToLink",
    category = SkylarkModuleCategory.BUILTIN,
    doc = "A library the user can link against.")
public interface LibraryToLinkApi<FileT extends FileApi> {
  @SkylarkCallable(
      name = "static_library",
      allowReturnNones = true,
      doc = "<code>Artifact</code> of static library to be linked.",
      structField = true)
  FileT getStaticLibrary();

  @SkylarkCallable(
      name = "pic_static_library",
      allowReturnNones = true,
      doc = "<code>Artifact</code> of pic static library to be linked.",
      structField = true)
  FileT getPicStaticLibrary();

  @SkylarkCallable(
      name = "dynamic_library",
      doc =
          "<code>Artifact</code> of dynamic library to be linked. Always used for runtime "
              + "and used for linking if <code>interface_library</code> is not passed.",
      allowReturnNones = true,
      structField = true)
  FileT getDynamicLibrary();

  @SkylarkCallable(
      name = "interface_library",
      doc = "<code>Artifact</code> of interface library to be linked.",
      allowReturnNones = true,
      structField = true)
  FileT getInterfaceLibrary();

  @SkylarkCallable(
      name = "alwayslink",
      doc = "Whether to link the static library/objects in the --whole_archive block.",
      allowReturnNones = true,
      structField = true)
  boolean getAlwayslink();
}
