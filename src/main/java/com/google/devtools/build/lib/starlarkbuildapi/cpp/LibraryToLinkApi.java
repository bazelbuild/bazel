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

package com.google.devtools.build.lib.starlarkbuildapi.cpp;

import com.google.devtools.build.docgen.annot.DocCategory;
import com.google.devtools.build.lib.starlarkbuildapi.FileApi;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.annot.StarlarkMethod;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkValue;

/**
 * A library the user can link to. This is different from a simple linker input in that it also has
 * a library identifier.
 */
@StarlarkBuiltin(
    name = "LibraryToLink",
    category = DocCategory.BUILTIN,
    doc = "A library the user can link against.")
public interface LibraryToLinkApi extends StarlarkValue {
  @StarlarkMethod(
      name = "objects",
      allowReturnNones = true,
      doc = "<code>List</code> of object files in the library.",
      structField = true)
  @Nullable
  Sequence<?> getObjectFilesForStarlark();

  @StarlarkMethod(
      name = "pic_objects",
      allowReturnNones = true,
      doc = "<code>List</code> of pic object files in the library.",
      structField = true)
  @Nullable
  Sequence<?> getPicObjectFilesForStarlark();

  @StarlarkMethod(
      name = "lto_bitcode_files",
      allowReturnNones = true,
      doc = "<code>List</code> of LTO bitcode files in the library.",
      structField = true)
  @Nullable
  Sequence<?> getLtoBitcodeFilesForStarlark();

  @StarlarkMethod(
      name = "pic_lto_bitcode_files",
      allowReturnNones = true,
      doc = "<code>List</code> of pic LTO bitcode files in the library.",
      structField = true)
  @Nullable
  Sequence<?> getPicLtoBitcodeFilesForStarlark();

  @StarlarkMethod(
      name = "static_library",
      allowReturnNones = true,
      doc = "<code>Artifact</code> of static library to be linked.",
      structField = true)
  @Nullable
  FileApi getStaticLibrary();

  @StarlarkMethod(
      name = "pic_static_library",
      allowReturnNones = true,
      doc = "<code>Artifact</code> of pic static library to be linked.",
      structField = true)
  @Nullable
  FileApi getPicStaticLibrary();

  @StarlarkMethod(
      name = "dynamic_library",
      doc =
          "<code>Artifact</code> of dynamic library to be linked. Always used for runtime "
              + "and used for linking if <code>interface_library</code> is not passed.",
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileApi getDynamicLibrary();

  @StarlarkMethod(
      name = "resolved_symlink_dynamic_library",
      doc =
          "The resolved <code>Artifact</code> of the dynamic library to be linked if "
              + "<code>dynamic_library</code> is a symlink, otherwise this is None.",
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileApi getResolvedSymlinkDynamicLibrary();

  @StarlarkMethod(
      name = "interface_library",
      doc = "<code>Artifact</code> of interface library to be linked.",
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileApi getInterfaceLibrary();

  @StarlarkMethod(
      name = "resolved_symlink_interface_library",
      doc =
          "The resolved <code>Artifact</code> of the interface library to be linked if "
              + "<code>interface_library</code> is a symlink, otherwise this is None.",
      allowReturnNones = true,
      structField = true)
  @Nullable
  FileApi getResolvedSymlinkInterfaceLibrary();

  @StarlarkMethod(
      name = "alwayslink",
      doc = "Whether to link the static library/objects in the --whole_archive block.",
      structField = true)
  boolean getAlwayslink();
}
