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

package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.collect.ImmutableList.toImmutableList;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;

/**
 * Unwraps information for linking a library from a Starlark struct.
 *
 * @deprecated Use only in tests
 */
@Deprecated
public final class LibraryToLink {
  @Deprecated
  public static ImmutableList<Artifact> getDynamicLibrariesForRuntime(
      boolean linkingStatically, Iterable<LibraryToLink> libraries) {
    ImmutableList.Builder<Artifact> dynamicLibrariesForRuntimeBuilder = ImmutableList.builder();
    for (LibraryToLink libraryToLink : libraries) {
      Artifact artifact = libraryToLink.getDynamicLibraryForRuntimeOrNull(linkingStatically);
      if (artifact != null) {
        dynamicLibrariesForRuntimeBuilder.add(artifact);
      }
    }
    return dynamicLibrariesForRuntimeBuilder.build();
  }

  @Deprecated
  public static ImmutableList<Artifact> getDynamicLibrariesForLinking(
      NestedSet<LibraryToLink> libraries) {
    ImmutableList.Builder<Artifact> dynamicLibrariesForLinkingBuilder = ImmutableList.builder();
    for (LibraryToLink libraryToLink : libraries.toList()) {
      if (libraryToLink.getInterfaceLibrary() != null) {
        dynamicLibrariesForLinkingBuilder.add(libraryToLink.getInterfaceLibrary());
      } else if (libraryToLink.getDynamicLibrary() != null) {
        dynamicLibrariesForLinkingBuilder.add(libraryToLink.getDynamicLibrary());
      }
    }
    return dynamicLibrariesForLinkingBuilder.build();
  }

  private final StarlarkInfo value;

  private LibraryToLink(StarlarkInfo value) {
    this.value = value;
  }

  public static LibraryToLink wrap(StarlarkInfo value) {
    return new LibraryToLink(value);
  }

  public static NestedSet<LibraryToLink> wrap(NestedSet<StarlarkInfo> libraries) {
    return NestedSetBuilder.wrap(
        Order.STABLE_ORDER,
        libraries.toList().stream().map(LibraryToLink::wrap).collect(toImmutableList()));
  }

  @Nullable
  public Artifact getStaticLibrary() {
    return value.getValue("static_library") instanceof Artifact artifact ? artifact : null;
  }

  @Nullable
  public Artifact getPicStaticLibrary() {
    return value.getValue("pic_static_library") instanceof Artifact artifact ? artifact : null;
  }

  @Nullable
  public Artifact getDynamicLibrary() {
    return value.getValue("dynamic_library") instanceof Artifact artifact ? artifact : null;
  }

  @Nullable
  public Artifact getResolvedSymlinkDynamicLibrary() {
    return value.getValue("resolved_symlink_dynamic_library") instanceof Artifact artifact
        ? artifact
        : null;
  }

  @Nullable
  public Artifact getInterfaceLibrary() {
    return value.getValue("interface_library") instanceof Artifact artifact ? artifact : null;
  }

  @Nullable
  public Artifact getResolvedSymlinkInterfaceLibrary() {
    return value.getValue("resolved_symlink_interface_library") instanceof Artifact artifact
        ? artifact
        : null;
  }

  @Nullable
  public ImmutableList<Artifact> getObjectFiles() throws EvalException {
    return Sequence.cast(value.getValue("objects"), Artifact.class, "objects").getImmutableList();
  }

  @Nullable
  public ImmutableList<Artifact> getPicObjectFiles() throws EvalException {
    return Sequence.cast(value.getValue("pic_objects"), Artifact.class, "pic_objects")
        .getImmutableList();
  }

  @Nullable
  public LtoCompilationContext getLtoCompilationContext() {
    return value.getValue("_lto_compilation_context") instanceof LtoCompilationContext ctx
        ? ctx
        : null;
  }

  @Nullable
  public LtoCompilationContext getPicLtoCompilationContext() {
    return value.getValue("_pic_lto_compilation_context") instanceof LtoCompilationContext ctx
        ? ctx
        : null;
  }

  public boolean getAlwayslink() {
    return value.getValue("alwayslink") instanceof Boolean bool && bool;
  }

  @Nullable
  public Artifact getDynamicLibraryForRuntimeOrNull(boolean linkingStatically) {
    if (getDynamicLibrary() == null) {
      return null;
    }
    if (linkingStatically && (getStaticLibrary() != null || getPicStaticLibrary() != null)) {
      return null;
    }
    return getDynamicLibrary();
  }
}
