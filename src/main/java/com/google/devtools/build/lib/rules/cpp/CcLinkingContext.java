// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.base.VerifyException;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcLinkingContextApi;
import java.util.List;
import net.starlark.java.eval.EvalException;

/** Helper class for accessing information from the CcLinkingContext provider. */
public class CcLinkingContext implements CcLinkingContextApi {
  private final StarlarkInfo ccLinkingContext;

  private CcLinkingContext(StarlarkInfo ccLinkingContext) {
    this.ccLinkingContext = ccLinkingContext;
  }

  public static CcLinkingContext of(StarlarkInfo ccLinkingContext) {
    return new CcLinkingContext(ccLinkingContext);
  }

  /**
   * Wraps any input to the linker, be it libraries, linker scripts, linkstamps or linking options.
   */
  public static final class LinkerInput {
    private LinkerInput() {}

    /**
     * @deprecated Use only in tests
     */
    @Deprecated
    public static Label getOwner(StarlarkInfo linkerInput) {
      try {
        return linkerInput.getValue("owner", Label.class);
      } catch (EvalException e) {
        throw new VerifyException(e);
      }
    }

    /**
     * @deprecated Use only in tests
     */
    @Deprecated
    public static ImmutableList<LibraryToLink> getLibraries(StarlarkInfo linkerInput) {
      try {
        @SuppressWarnings("unchecked")
        ImmutableList<LibraryToLink> libraries =
            ((List<StarlarkInfo>) linkerInput.getValue("libraries", List.class))
                .stream().map(LibraryToLink::wrap).collect(toImmutableList());
        return libraries;
      } catch (EvalException e) {
        throw new VerifyException(e);
      }
    }

    /**
     * @deprecated Use only in tests
     */
    @Deprecated
    public static List<String> getUserLinkFlags(StarlarkInfo linkerInput) {
      try {
        @SuppressWarnings("unchecked")
        List<String> userLinkFlags =
            (List<String>) linkerInput.getValue("user_link_flags", List.class);
        return userLinkFlags;
      } catch (EvalException e) {
        throw new VerifyException(e);
      }
    }

    /**
     * @deprecated Use only in tests
     */
    @Deprecated
    public static List<Artifact> getNonCodeInputs(StarlarkInfo linkerInput) throws EvalException {
      @SuppressWarnings("unchecked")
      List<Artifact> additionalInputs =
          (List<Artifact>) linkerInput.getValue("additional_inputs", List.class);
      return additionalInputs;
    }
  }

  /**
   * @deprecated Only use in tests
   */
  @Deprecated
  public List<Artifact> getStaticModeParamsForExecutableLibraries() throws EvalException {
    ImmutableList.Builder<Artifact> libraryListBuilder = ImmutableList.builder();
    for (LibraryToLink libraryToLink : getLibraries().toList()) {
      if (libraryToLink.getStaticLibrary() != null) {
        libraryListBuilder.add(libraryToLink.getStaticLibrary());
      } else if (libraryToLink.getPicStaticLibrary() != null) {
        libraryListBuilder.add(libraryToLink.getPicStaticLibrary());
      } else if (libraryToLink.getInterfaceLibrary() != null) {
        libraryListBuilder.add(libraryToLink.getInterfaceLibrary());
      } else {
        libraryListBuilder.add(libraryToLink.getDynamicLibrary());
      }
    }
    return libraryListBuilder.build();
  }

  /**
   * @deprecated Only use in tests
   */
  @Deprecated
  public List<Artifact> getStaticModeParamsForDynamicLibraryLibraries() throws EvalException {
    ImmutableList.Builder<Artifact> artifactListBuilder = ImmutableList.builder();
    for (LibraryToLink library : getLibraries().toList()) {
      if (library.getPicStaticLibrary() != null) {
        artifactListBuilder.add(library.getPicStaticLibrary());
      } else if (library.getStaticLibrary() != null) {
        artifactListBuilder.add(library.getStaticLibrary());
      } else if (library.getInterfaceLibrary() != null) {
        artifactListBuilder.add(library.getInterfaceLibrary());
      } else {
        artifactListBuilder.add(library.getDynamicLibrary());
      }
    }
    return artifactListBuilder.build();
  }

  /**
   * @deprecated Use only in tests. @Deprecated
   */
  @Deprecated
  public List<Artifact> getDynamicLibrariesForRuntime(boolean linkingStatically)
      throws EvalException {
    return LibraryToLink.getDynamicLibrariesForRuntime(linkingStatically, getLibraries().toList());
  }

  /**
   * @deprecated Use only in tests
   */
  @Deprecated
  public NestedSet<LibraryToLink> getLibraries() throws EvalException {
    NestedSetBuilder<LibraryToLink> libraries = NestedSetBuilder.linkOrder();
    for (StarlarkInfo linkerInput : getLinkerInputs().toList()) {
      libraries.addAll(LinkerInput.getLibraries(linkerInput));
    }
    return libraries.build();
  }

  /**
   * @deprecated Use only in tests
   */
  @Deprecated
  public NestedSet<StarlarkInfo> getLinkerInputs() {
    try {
      return Depset.cast(
          ccLinkingContext.getValue("linker_inputs"), StarlarkInfo.class, "linker_inputs");
    } catch (EvalException e) {
      throw new VerifyException(e);
    }
  }

  /**
   * @deprecated Only use in tests. Inline, using LinkerInputs.
   */
  @Deprecated
  public ImmutableList<String> getFlattenedUserLinkFlags() {
    return getLinkerInputs().toList().stream()
        .flatMap(linkerInput -> LinkerInput.getUserLinkFlags(linkerInput).stream())
        .collect(toImmutableList());
  }

  /**
   * @deprecated Only use in tests. Inline, using LinkerInputs.
   */
  @Deprecated
  public NestedSet<Artifact> getNonCodeInputs() throws EvalException {
    NestedSetBuilder<Artifact> nonCodeInputs = NestedSetBuilder.linkOrder();
    for (StarlarkInfo linkerInput : getLinkerInputs().toList()) {
      nonCodeInputs.addAll(LinkerInput.getNonCodeInputs(linkerInput));
    }
    return nonCodeInputs.build();
  }
}
