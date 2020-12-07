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

import com.google.common.base.Joiner;
import com.google.common.base.MoreObjects;
import com.google.common.base.Objects;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Streams;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.Depset;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.packages.SymbolGenerator;
import com.google.devtools.build.lib.packages.semantics.BuildLanguageOptions;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcLinkingContextApi;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.LinkerInputApi;
import com.google.devtools.build.lib.util.Fingerprint;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;

/** Structure of CcLinkingContext. */
public class CcLinkingContext implements CcLinkingContextApi<Artifact> {
  public static final CcLinkingContext EMPTY = builder().build();

  /** A list of link options contributed by a single configured target/aspect. */
  @Immutable
  public static final class LinkOptions {
    private final ImmutableList<String> linkOptions;
    private final Object symbolForEquality;

    private LinkOptions(ImmutableList<String> linkOptions, Object symbolForEquality) {
      this.linkOptions = Preconditions.checkNotNull(linkOptions);
      this.symbolForEquality = Preconditions.checkNotNull(symbolForEquality);
    }

    public ImmutableList<String> get() {
      return linkOptions;
    }

    public static LinkOptions of(
        ImmutableList<String> linkOptions, SymbolGenerator<?> symbolGenerator) {
      return new LinkOptions(linkOptions, symbolGenerator.generate());
    }

    @Override
    public int hashCode() {
      // Symbol is sufficient for equality check.
      return symbolForEquality.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof LinkOptions)) {
        return false;
      }
      LinkOptions that = (LinkOptions) obj;
      if (!this.symbolForEquality.equals(that.symbolForEquality)) {
        return false;
      }
      if (this.linkOptions.equals(that.linkOptions)) {
        return true;
      }
      BugReport.sendBugReport(
          new IllegalStateException(
              "Unexpected inequality with equal symbols: " + this + ", " + that));
      return false;
    }

    @Override
    public String toString() {
      return '[' + Joiner.on(",").join(linkOptions) + "] (owner: " + symbolForEquality;
    }
  }

  /**
   * A linkstamp that also knows about its declared includes.
   *
   * <p>This object is required because linkstamp files may include other headers which will have to
   * be provided during compilation.
   */
  public static final class Linkstamp {
    private final Artifact artifact;
    private final NestedSet<Artifact> declaredIncludeSrcs;
    private final byte[] nestedDigest;

    // TODO(janakr): if action key context is not available, the digest can be computed lazily,
    // only if we are doing an equality comparison and artifacts are equal. That should never
    // happen, so doing an expensive digest should be ok then. If this is ever moved to Starlark
    // and Starlark doesn't support custom equality or amortized deep equality of nested sets, a
    // Symbol can be used as an equality proxy, similar to what LinkOptions does above.
    Linkstamp(
        Artifact artifact,
        NestedSet<Artifact> declaredIncludeSrcs,
        ActionKeyContext actionKeyContext)
        throws CommandLineExpansionException, InterruptedException {
      this.artifact = Preconditions.checkNotNull(artifact);
      this.declaredIncludeSrcs = Preconditions.checkNotNull(declaredIncludeSrcs);
      Fingerprint fp = new Fingerprint();
      actionKeyContext.addNestedSetToFingerprint(fp, this.declaredIncludeSrcs);
      nestedDigest = fp.digestAndReset();
    }

    /** Returns the linkstamp artifact. */
    public Artifact getArtifact() {
      return artifact;
    }

    /** Returns the declared includes. */
    public NestedSet<Artifact> getDeclaredIncludeSrcs() {
      return declaredIncludeSrcs;
    }

    @Override
    public int hashCode() {
      // Artifact should be enough to disambiguate basically all the time.
      return artifact.hashCode();
    }

    @Override
    public boolean equals(Object obj) {
      if (this == obj) {
        return true;
      }
      if (!(obj instanceof Linkstamp)) {
        return false;
      }
      Linkstamp other = (Linkstamp) obj;
      return artifact.equals(other.artifact)
          && Arrays.equals(this.nestedDigest, other.nestedDigest);
    }
  }

  /**
   * Wraps any input to the linker, be it libraries, linker scripts, linkstamps or linking options.
   */
  // TODO(bazel-team): choose less confusing names for this class and the package-level interface of
  // the same name.
  @Immutable
  public static class LinkerInput
      implements LinkerInputApi<LibraryToLink, LtoBackendArtifacts, Artifact> {

    public static final Depset.ElementType TYPE = Depset.ElementType.of(LinkerInput.class);

    // Identifies which target created the LinkerInput. It doesn't have to be unique between
    // LinkerInputs.
    private final Label owner;
    private final ImmutableList<LibraryToLink> libraries;
    private final ImmutableList<LinkOptions> userLinkFlags;
    private final ImmutableList<Artifact> nonCodeInputs;
    private final ImmutableList<Linkstamp> linkstamps;

    private LinkerInput(
        Label owner,
        ImmutableList<LibraryToLink> libraries,
        ImmutableList<LinkOptions> userLinkFlags,
        ImmutableList<Artifact> nonCodeInputs,
        ImmutableList<Linkstamp> linkstamps) {
      this.owner = owner;
      this.libraries = libraries;
      this.userLinkFlags = userLinkFlags;
      this.nonCodeInputs = nonCodeInputs;
      this.linkstamps = linkstamps;
    }

    @Override
    public boolean isImmutable() {
      return true; // immutable and Starlark-hashable
    }

    @Override
    public Label getStarlarkOwner() throws EvalException {
      if (owner == null) {
        throw Starlark.errorf(
            "Owner is null. This means that some target upstream is of a rule type that uses the"
                + " old API of create_linking_context");
      }
      return owner;
    }

    public Label getOwner() {
      return owner;
    }

    public List<LibraryToLink> getLibraries() {
      return libraries;
    }

    @Override
    public Sequence<LibraryToLink> getStarlarkLibrariesToLink(StarlarkSemantics semantics) {
      return StarlarkList.immutableCopyOf(getLibraries());
    }

    public List<LinkOptions> getUserLinkFlags() {
      return userLinkFlags;
    }

    @Override
    public Sequence<String> getStarlarkUserLinkFlags() {
      return StarlarkList.immutableCopyOf(
          getUserLinkFlags().stream()
              .map(LinkOptions::get)
              .flatMap(Collection::stream)
              .collect(ImmutableList.toImmutableList()));
    }

    public List<Artifact> getNonCodeInputs() {
      return nonCodeInputs;
    }

    @Override
    public Sequence<Artifact> getStarlarkNonCodeInputs() {
      return StarlarkList.immutableCopyOf(getNonCodeInputs());
    }

    public List<Linkstamp> getLinkstamps() {
      return linkstamps;
    }

    @Override
    public void debugPrint(Printer printer) {
      printer.append("<LinkerInput(owner=");
      owner.debugPrint(printer);
      printer.append(", libraries=[");
      for (LibraryToLink libraryToLink : libraries) {
        libraryToLink.debugPrint(printer);
        printer.append(", ");
      }
      printer.append("], userLinkFlags=[");
      printer.append(Joiner.on(", ").join(userLinkFlags));
      printer.append("], nonCodeInputs=[");
      for (Artifact nonCodeInput : nonCodeInputs) {
        nonCodeInput.debugPrint(printer);
        printer.append(", ");
      }
      // TODO(cparsons): Add debug repesentation of linkstamps.
      printer.append("])>");
    }

    public static Builder builder() {
      return new Builder();
    }

    /** Builder for {@link LinkerInput} */
    public static class Builder {
      private Label owner;
      private final ImmutableList.Builder<LibraryToLink> libraries = ImmutableList.builder();
      private final ImmutableList.Builder<LinkOptions> userLinkFlags = ImmutableList.builder();
      private final ImmutableList.Builder<Artifact> nonCodeInputs = ImmutableList.builder();
      private final ImmutableList.Builder<Linkstamp> linkstamps = ImmutableList.builder();

      public Builder addLibrary(LibraryToLink library) {
        this.libraries.add(library);
        return this;
      }

      public Builder addLibraries(List<LibraryToLink> libraries) {
        this.libraries.addAll(libraries);
        return this;
      }

      public Builder addUserLinkFlags(List<LinkOptions> userLinkFlags) {
        this.userLinkFlags.addAll(userLinkFlags);
        return this;
      }

      public Builder addLinkstamps(List<Linkstamp> linkstamps) {
        this.linkstamps.addAll(linkstamps);
        return this;
      }

      public Builder addNonCodeInputs(List<Artifact> nonCodeInputs) {
        this.nonCodeInputs.addAll(nonCodeInputs);
        return this;
      }

      public Builder setOwner(Label owner) {
        this.owner = owner;
        return this;
      }

      public LinkerInput build() {
        return new LinkerInput(
            owner,
            libraries.build(),
            userLinkFlags.build(),
            nonCodeInputs.build(),
            linkstamps.build());
      }
    }

    @Override
    public boolean equals(Object otherObject) {
      if (!(otherObject instanceof LinkerInput)) {
        return false;
      }
      LinkerInput other = (LinkerInput) otherObject;
      if (this == other) {
        return true;
      }
      return Objects.equal(this.owner, other.owner)
          && this.libraries.equals(other.libraries)
          && this.userLinkFlags.equals(other.userLinkFlags)
          && this.linkstamps.equals(other.linkstamps)
          && this.nonCodeInputs.equals(other.nonCodeInputs);
    }

    @Override
    public int hashCode() {
      return Objects.hashCode(
          libraries.hashCode(),
          userLinkFlags.hashCode(),
          linkstamps.hashCode(),
          nonCodeInputs.hashCode());
    }

    @Override
    public String toString() {
      return MoreObjects.toStringHelper(this)
          .add("userLinkFlags", userLinkFlags)
          .add("linkstamps", linkstamps)
          .add("libraries", libraries)
          .add("nonCodeInputs", nonCodeInputs)
          .toString();
    }
  }

  private final NestedSet<LinkerInput> linkerInputs;
  private final ExtraLinkTimeLibraries extraLinkTimeLibraries;

  @Override
  public void debugPrint(Printer printer) {
    printer.append("<CcLinkingContext([");
    for (LinkerInput linkerInput : linkerInputs.toList()) {
      linkerInput.debugPrint(printer);
      printer.append(", ");
    }
    printer.append("])>");
  }

  public CcLinkingContext(
      NestedSet<LinkerInput> linkerInputs, ExtraLinkTimeLibraries extraLinkTimeLibraries) {
    this.linkerInputs = linkerInputs;
    this.extraLinkTimeLibraries = extraLinkTimeLibraries;
  }

  public static CcLinkingContext merge(List<CcLinkingContext> ccLinkingContexts) {
    Builder mergedCcLinkingContext = CcLinkingContext.builder();
    ExtraLinkTimeLibraries.Builder mergedExtraLinkTimeLibraries = ExtraLinkTimeLibraries.builder();
    for (CcLinkingContext ccLinkingContext : ccLinkingContexts) {
      mergedCcLinkingContext.addTransitiveLinkerInputs(ccLinkingContext.getLinkerInputs());
      if (ccLinkingContext.getExtraLinkTimeLibraries() != null) {
        mergedExtraLinkTimeLibraries.addTransitive(ccLinkingContext.getExtraLinkTimeLibraries());
      }
    }
    mergedCcLinkingContext.setExtraLinkTimeLibraries(mergedExtraLinkTimeLibraries.build());
    return mergedCcLinkingContext.build();
  }

  public List<Artifact> getStaticModeParamsForExecutableLibraries() {
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

  public List<Artifact> getStaticModeParamsForDynamicLibraryLibraries() {
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

  public List<Artifact> getDynamicLibrariesForRuntime(boolean linkingStatically) {
    return LibraryToLink.getDynamicLibrariesForRuntime(linkingStatically, getLibraries().toList());
  }

  public NestedSet<LibraryToLink> getLibraries() {
    NestedSetBuilder<LibraryToLink> libraries = NestedSetBuilder.linkOrder();
    for (LinkerInput linkerInput : linkerInputs.toList()) {
      libraries.addAll(linkerInput.libraries);
    }
    return libraries.build();
  }

  public NestedSet<LinkerInput> getLinkerInputs() {
    return linkerInputs;
  }

  @Override
  public Depset getStarlarkLinkerInputs() {
    return Depset.of(LinkerInput.TYPE, linkerInputs);
  }

  @Override
  public Sequence<String> getStarlarkUserLinkFlags() {
    return StarlarkList.immutableCopyOf(getFlattenedUserLinkFlags());
  }

  @Override
  public Object getStarlarkLibrariesToLink(StarlarkSemantics semantics) {
    // TODO(plf): Flag can be removed already.
    if (semantics.getBool(BuildLanguageOptions.INCOMPATIBLE_DEPSET_FOR_LIBRARIES_TO_LINK_GETTER)) {
      return Depset.of(LibraryToLink.TYPE, getLibraries());
    } else {
      return StarlarkList.immutableCopyOf(getLibraries().toList());
    }
  }

  @Override
  public Depset getStarlarkNonCodeInputs() {
    return Depset.of(Artifact.TYPE, getNonCodeInputs());
  }

  public NestedSet<LinkOptions> getUserLinkFlags() {
    NestedSetBuilder<LinkOptions> userLinkFlags = NestedSetBuilder.linkOrder();
    for (LinkerInput linkerInput : linkerInputs.toList()) {
      userLinkFlags.addAll(linkerInput.getUserLinkFlags());
    }
    return userLinkFlags.build();
  }

  public ImmutableList<String> getFlattenedUserLinkFlags() {
    return Streams.stream(getUserLinkFlags().toList())
        .map(LinkOptions::get)
        .flatMap(Collection::stream)
        .collect(ImmutableList.toImmutableList());
  }

  public NestedSet<Linkstamp> getLinkstamps() {
    NestedSetBuilder<Linkstamp> linkstamps = NestedSetBuilder.linkOrder();
    for (LinkerInput linkerInput : linkerInputs.toList()) {
      linkstamps.addAll(linkerInput.getLinkstamps());
    }
    return linkstamps.build();
  }

  public NestedSet<Artifact> getNonCodeInputs() {
    NestedSetBuilder<Artifact> nonCodeInputs = NestedSetBuilder.linkOrder();
    for (LinkerInput linkerInput : linkerInputs.toList()) {
      nonCodeInputs.addAll(linkerInput.getNonCodeInputs());
    }
    return nonCodeInputs.build();
  }

  public ExtraLinkTimeLibraries getExtraLinkTimeLibraries() {
    return extraLinkTimeLibraries;
  }

  public static Builder builder() {
    // private to avoid class initialization deadlock between this class and its outer class
    return new Builder();
  }

  /** Builder for {@link CcLinkingContext}. */
  public static class Builder {
    boolean hasDirectLinkerInput;
    LinkerInput.Builder linkerInputBuilder = LinkerInput.builder();
    private final NestedSetBuilder<LinkerInput> linkerInputs = NestedSetBuilder.linkOrder();
    private ExtraLinkTimeLibraries extraLinkTimeLibraries = null;

    public Builder setOwner(Label owner) {
      linkerInputBuilder.setOwner(owner);
      return this;
    }

    public Builder addLibrary(LibraryToLink library) {
      hasDirectLinkerInput = true;
      linkerInputBuilder.addLibrary(library);
      return this;
    }

    public Builder addLibraries(List<LibraryToLink> libraries) {
      hasDirectLinkerInput = true;
      linkerInputBuilder.addLibraries(libraries);
      return this;
    }

    public Builder addUserLinkFlags(List<LinkOptions> userLinkFlags) {
      hasDirectLinkerInput = true;
      linkerInputBuilder.addUserLinkFlags(userLinkFlags);
      return this;
    }

    Builder addLinkstamps(List<Linkstamp> linkstamps) {
      hasDirectLinkerInput = true;
      linkerInputBuilder.addLinkstamps(linkstamps);
      return this;
    }

    Builder addNonCodeInputs(List<Artifact> nonCodeInputs) {
      hasDirectLinkerInput = true;
      linkerInputBuilder.addNonCodeInputs(nonCodeInputs);
      return this;
    }

    public Builder addTransitiveLinkerInputs(NestedSet<LinkerInput> linkerInputs) {
      this.linkerInputs.addTransitive(linkerInputs);
      return this;
    }

    public Builder setExtraLinkTimeLibraries(ExtraLinkTimeLibraries extraLinkTimeLibraries) {
      Preconditions.checkState(this.extraLinkTimeLibraries == null);
      this.extraLinkTimeLibraries = extraLinkTimeLibraries;
      return this;
    }

    public CcLinkingContext build() {
      if (hasDirectLinkerInput) {
        linkerInputs.add(linkerInputBuilder.build());
      }
      return new CcLinkingContext(linkerInputs.build(), extraLinkTimeLibraries);
    }
  }

  @Override
  public boolean equals(Object otherObject) {
    if (!(otherObject instanceof CcLinkingContext)) {
      return false;
    }
    CcLinkingContext other = (CcLinkingContext) otherObject;
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
