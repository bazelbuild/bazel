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

import com.google.common.base.MoreObjects;
import com.google.common.base.Preconditions;
import com.google.common.base.VerifyException;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcLinkingContextApi;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.StarlarkThread;

/** Structure of CcLinkingContext. */
public class CcLinkingContext implements CcLinkingContextApi<Artifact> {
  public static final CcLinkingContext EMPTY =
      builder().setExtraLinkTimeLibraries(ExtraLinkTimeLibraries.EMPTY).build();

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

  private final NestedSet<StarlarkInfo> linkerInputs;
  @Nullable private final ExtraLinkTimeLibraries extraLinkTimeLibraries;

  @Override
  public void debugPrint(Printer printer, StarlarkThread thread) {
    printer.append("<CcLinkingContext([");
    for (StarlarkInfo linkerInput : linkerInputs.toList()) {
      linkerInput.debugPrint(printer, thread);
      printer.append(", ");
    }
    printer.append("])>");
  }

  public CcLinkingContext(
      NestedSet<StarlarkInfo> linkerInputs,
      @Nullable ExtraLinkTimeLibraries extraLinkTimeLibraries) {
    this.linkerInputs = linkerInputs;
    this.extraLinkTimeLibraries = extraLinkTimeLibraries;
  }

  public static CcLinkingContext merge(List<CcLinkingContext> ccLinkingContexts) {
    if (ccLinkingContexts.isEmpty()) {
      return EMPTY;
    }
    Builder mergedCcLinkingContext = CcLinkingContext.builder();
    ImmutableList.Builder<ExtraLinkTimeLibraries> extraLinkTimeLibrariesBuilder =
        ImmutableList.builder();
    for (CcLinkingContext ccLinkingContext : ccLinkingContexts) {
      mergedCcLinkingContext.addTransitiveLinkerInputs(ccLinkingContext.getLinkerInputs());
      if (ccLinkingContext.getExtraLinkTimeLibraries() != null) {
        extraLinkTimeLibrariesBuilder.add(ccLinkingContext.getExtraLinkTimeLibraries());
      }
    }
    mergedCcLinkingContext.setExtraLinkTimeLibraries(
        ExtraLinkTimeLibraries.merge(extraLinkTimeLibrariesBuilder.build()));
    return mergedCcLinkingContext.build();
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
    for (StarlarkInfo linkerInput : linkerInputs.toList()) {
      libraries.addAll(LinkerInput.getLibraries(linkerInput));
    }
    return libraries.build();
  }

  public NestedSet<StarlarkInfo> getLinkerInputs() {
    return linkerInputs;
  }

  @Override
  public Depset getStarlarkLinkerInputs() {
    return Depset.of(StarlarkInfo.class, linkerInputs);
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
    for (StarlarkInfo linkerInput : linkerInputs.toList()) {
      nonCodeInputs.addAll(LinkerInput.getNonCodeInputs(linkerInput));
    }
    return nonCodeInputs.build();
  }

  public ExtraLinkTimeLibraries getExtraLinkTimeLibraries() {
    return extraLinkTimeLibraries;
  }

  @Override
  public ExtraLinkTimeLibraries getExtraLinkTimeLibrariesForStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return getExtraLinkTimeLibraries();
  }

  public static Builder builder() {
    // private to avoid class initialization deadlock between this class and its outer class
    return new Builder();
  }

  /** Builder for {@link CcLinkingContext}. */
  public static class Builder {
    private final NestedSetBuilder<StarlarkInfo> linkerInputs = NestedSetBuilder.linkOrder();
    private ExtraLinkTimeLibraries extraLinkTimeLibraries = null;

    @CanIgnoreReturnValue
    public Builder addTransitiveLinkerInputs(NestedSet<StarlarkInfo> linkerInputs) {
      this.linkerInputs.addTransitive(linkerInputs);
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExtraLinkTimeLibraries(ExtraLinkTimeLibraries extraLinkTimeLibraries) {
      Preconditions.checkState(this.extraLinkTimeLibraries == null);
      this.extraLinkTimeLibraries = extraLinkTimeLibraries;
      return this;
    }

    public CcLinkingContext build() {
      return new CcLinkingContext(linkerInputs.build(), extraLinkTimeLibraries);
    }
  }

  @Override
  public boolean equals(Object otherObject) {
    if (!(otherObject instanceof CcLinkingContext other)) {
      return false;
    }
    if (this == other) {
      return true;
    }
    return this.linkerInputs.shallowEquals(other.linkerInputs);
  }

  @Override
  public int hashCode() {
    return linkerInputs.shallowHashCode();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this).add("linkerInputs", linkerInputs).toString();
  }
}
