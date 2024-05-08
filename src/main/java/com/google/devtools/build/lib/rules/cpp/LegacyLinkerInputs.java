// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ThreadSafety;

/** Factory for creating new {@link LegacyLinkerInput} objects. */
public abstract class LegacyLinkerInputs {
  /**
   * An opaque linker input that is not a library, for example a linker script or an individual
   * object file.
   */
  @ThreadSafety.Immutable
  private static class SimpleLinkerInput implements LegacyLinkerInput {
    private final Artifact artifact;
    private final ArtifactCategory category;
    private final boolean disableWholeArchive;
    private final String libraryIdentifier;

    SimpleLinkerInput(
        Artifact artifact,
        ArtifactCategory category,
        boolean disableWholeArchive,
        String libraryIdentifier) {
      Preconditions.checkNotNull(libraryIdentifier);
      String basename = artifact.getFilename();
      switch (category) {
        case STATIC_LIBRARY:
          Preconditions.checkState(Link.ARCHIVE_LIBRARY_FILETYPES.matches(basename));
          break;

        case DYNAMIC_LIBRARY:
          Preconditions.checkState(Link.SHARED_LIBRARY_FILETYPES.matches(basename));
          break;

        case OBJECT_FILE:
          // We skip file extension checks for TreeArtifacts because they represent directory
          // artifacts without a file extension.
          Preconditions.checkState(
              artifact.isTreeArtifact() || Link.OBJECT_FILETYPES.matches(basename));
          break;

        default:
          throw new IllegalStateException();
      }
      this.artifact = Preconditions.checkNotNull(artifact);
      this.category = category;
      this.disableWholeArchive = disableWholeArchive;
      this.libraryIdentifier = libraryIdentifier;
    }

    @Override
    public ArtifactCategory getArtifactCategory() {
      return category;
    }

    @Override
    public Artifact getArtifact() {
      return artifact;
    }

    @Override
    public Artifact getOriginalLibraryArtifact() {
      return artifact;
    }

    @Override
    public boolean containsObjectFiles() {
      return false;
    }

    @Override
    public ImmutableCollection<Artifact> getObjectFiles() {
      throw new IllegalStateException();
    }

    @Override
    public boolean equals(Object that) {
      if (this == that) {
        return true;
      }

      if (!(that instanceof SimpleLinkerInput other)) {
        return false;
      }

      return artifact.equals(other.artifact);
    }

    @Override
    public int hashCode() {
      return artifact.hashCode();
    }

    @Override
    public String toString() {
      return "SimpleLinkerInput(" + artifact + ")";
    }

    @Override
    public boolean isMustKeepDebug() {
      return false;
    }

    @Override
    public boolean disableWholeArchive() {
      return disableWholeArchive;
    }

    @Override
    public String getLibraryIdentifier() {
      return libraryIdentifier;
    }
  }

  @ThreadSafety.Immutable
  private static class LinkstampLinkerInput extends SimpleLinkerInput {
    private LinkstampLinkerInput(Artifact artifact, String libraryIdentifier) {
      super(
          artifact,
          ArtifactCategory.OBJECT_FILE,
          /* disableWholeArchive= */ false,
          libraryIdentifier);
      Preconditions.checkState(Link.OBJECT_FILETYPES.matches(artifact.getFilename()));
    }

    @Override
    public boolean isLinkstamp() {
      return true;
    }
  }

  /**
   * A library the user can link to. This is different from a simple linker input in that it also
   * has a library identifier.
   */
  public interface LibraryInput extends LegacyLinkerInput {
    LtoCompilationContext getLtoCompilationContext();

    /**
     * Return a map of object file artifacts to associated LTOBackendArtifacts objects generated
     * when LTO backend actions are to be shared among different targets using this library. This is
     * the case when we opt not to perform the LTO indexing step, such as when building tests with
     * static linking. ThinLTO is otherwise too expensive when statically linking tests, due to the
     * number of LTO backends that can be generated for a single blaze test invocation.
     */
    ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends();
  }

  /**
   * This class represents a solib library symlink. Its library identifier is inherited from the
   * library that it links to.
   */
  @ThreadSafety.Immutable
  private static class SolibLibraryInput implements LibraryInput {
    private final Artifact solibSymlinkArtifact;
    private final Artifact libraryArtifact;
    private final String libraryIdentifier;

    SolibLibraryInput(
        Artifact solibSymlinkArtifact, Artifact libraryArtifact, String libraryIdentifier) {
      Preconditions.checkArgument(
          Link.SHARED_LIBRARY_FILETYPES.matches(solibSymlinkArtifact.getFilename()));
      this.solibSymlinkArtifact = solibSymlinkArtifact;
      this.libraryArtifact = libraryArtifact;
      this.libraryIdentifier = libraryIdentifier;
    }

    @Override
    public String toString() {
      return String.format("SolibLibraryInput(%s -> %s", solibSymlinkArtifact, libraryArtifact);
    }

    @Override
    public ArtifactCategory getArtifactCategory() {
      return ArtifactCategory.DYNAMIC_LIBRARY;
    }

    @Override
    public Artifact getArtifact() {
      return solibSymlinkArtifact;
    }

    @Override
    public String getLibraryIdentifier() {
      return libraryIdentifier;
    }

    @Override
    public boolean containsObjectFiles() {
      return false;
    }

    @Override
    public LtoCompilationContext getLtoCompilationContext() {
      return LtoCompilationContext.EMPTY;
    }

    @Override
    public ImmutableCollection<Artifact> getObjectFiles() {
      throw new IllegalStateException(
          "LegacyLinkerInputs: does not support getObjectFiles: " + this);
    }

    @Override
    public ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends() {
      throw new IllegalStateException(
          "LegacyLinkerInputs: does not support getSharedNonLtoBackends: " + this);
    }

    @Override
    public Artifact getOriginalLibraryArtifact() {
      return libraryArtifact;
    }

    @Override
    public boolean equals(Object that) {
      if (this == that) {
        return true;
      }

      if (!(that instanceof SolibLibraryInput thatSolib)) {
        return false;
      }

      return solibSymlinkArtifact.equals(thatSolib.solibSymlinkArtifact)
          && libraryArtifact.equals(thatSolib.libraryArtifact);
    }

    @Override
    public int hashCode() {
      return solibSymlinkArtifact.hashCode();
    }

    @Override
    public boolean isMustKeepDebug() {
      return false;
    }

    @Override
    public boolean disableWholeArchive() {
      return false;
    }
  }

  /** This class represents a library that may contain object files. */
  @ThreadSafety.Immutable
  private static class CompoundLibraryInput implements LibraryInput {
    private final Artifact libraryArtifact;
    private final ArtifactCategory category;
    private final String libraryIdentifier;
    private final ImmutableCollection<Artifact> objectFiles;
    private final LtoCompilationContext ltoCompilationContext;
    private final ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends;
    private final boolean mustKeepDebug;
    private final boolean disableWholeArchive;

    CompoundLibraryInput(
        Artifact libraryArtifact,
        ArtifactCategory category,
        String libraryIdentifier,
        ImmutableCollection<Artifact> objectFiles,
        LtoCompilationContext ltoCompilationContext,
        ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends,
        boolean allowArchiveTypeInAlwayslink,
        boolean mustKeepDebug,
        boolean disableWholeArchive) {
      String basename = libraryArtifact.getFilename();
      switch (category) {
        case ALWAYSLINK_STATIC_LIBRARY:
          Preconditions.checkState(
              Link.LINK_LIBRARY_FILETYPES.matches(basename)
                  || (allowArchiveTypeInAlwayslink && Link.ARCHIVE_FILETYPES.matches(basename)));
          break;

        case STATIC_LIBRARY:
          Preconditions.checkState(Link.ARCHIVE_FILETYPES.matches(basename));
          break;

        case INTERFACE_LIBRARY:
        case DYNAMIC_LIBRARY:
          Preconditions.checkState(Link.SHARED_LIBRARY_FILETYPES.matches(basename));
          break;

        default:
          throw new IllegalStateException();
      }

      this.libraryArtifact = Preconditions.checkNotNull(libraryArtifact);
      this.category = category;
      this.libraryIdentifier = libraryIdentifier;
      this.objectFiles = objectFiles;
      this.ltoCompilationContext =
          (ltoCompilationContext == null) ? LtoCompilationContext.EMPTY : ltoCompilationContext;
      this.sharedNonLtoBackends = sharedNonLtoBackends;
      this.mustKeepDebug = mustKeepDebug;
      this.disableWholeArchive = disableWholeArchive;
    }

    @Override
    public String toString() {
      return String.format("CompoundLibraryInput(%s)", libraryArtifact);
    }

    @Override
    public ArtifactCategory getArtifactCategory() {
      return category;
    }

    @Override
    public Artifact getArtifact() {
      return libraryArtifact;
    }

    @Override
    public Artifact getOriginalLibraryArtifact() {
      return libraryArtifact;
    }

    @Override
    public String getLibraryIdentifier() {
      return libraryIdentifier;
    }

    @Override
    public boolean containsObjectFiles() {
      return objectFiles != null;
    }

    @Override
    public ImmutableMap<Artifact, LtoBackendArtifacts> getSharedNonLtoBackends() {
      return sharedNonLtoBackends;
    }

    @Override
    public ImmutableCollection<Artifact> getObjectFiles() {
      return Preconditions.checkNotNull(objectFiles);
    }

    @Override
    public LtoCompilationContext getLtoCompilationContext() {
      return ltoCompilationContext;
    }

    @Override
    public boolean equals(Object that) {
      if (this == that) {
        return true;
      }

      if (!(that instanceof CompoundLibraryInput)) {
        return false;
      }

      return libraryArtifact.equals(((CompoundLibraryInput) that).libraryArtifact);
    }

    @Override
    public int hashCode() {
      return libraryArtifact.hashCode();
    }

    @Override
    public boolean isMustKeepDebug() {
      return this.mustKeepDebug;
    }

    @Override
    public boolean disableWholeArchive() {
      return disableWholeArchive;
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////
  // Public factory constructors:
  //////////////////////////////////////////////////////////////////////////////////////

  /** Creates linker input objects for non-library files. */
  public static Iterable<LegacyLinkerInput> simpleLinkerInputs(
      Iterable<Artifact> input, final ArtifactCategory category, boolean disableWholeArchive) {
    return Iterables.transform(
        input,
        artifact ->
            simpleLinkerInput(
                artifact, category, disableWholeArchive, artifact.getRootRelativePathString()));
  }

  public static Iterable<LegacyLinkerInput> linkstampLinkerInputs(Iterable<Artifact> input) {
    return Iterables.transform(
        input,
        artifact -> new LinkstampLinkerInput(artifact, artifact.getRootRelativePathString()));
  }

  public static LegacyLinkerInput linkstampLinkerInput(Artifact input) {
    return new LinkstampLinkerInput(input, input.getRootRelativePathString());
  }

  /** Creates a linker input for which we do not know what objects files it consists of. */
  public static LegacyLinkerInput simpleLinkerInput(
      Artifact artifact,
      ArtifactCategory category,
      boolean disableWholeArchive,
      String libraryIdentifier) {
    // This precondition check was in place and *most* of the tests passed with them; the only
    // exception is when you mention a generated .a file in the srcs of a cc_* rule.
    // Preconditions.checkArgument(!ARCHIVE_LIBRARY_FILETYPES.contains(artifact.getFileType()));
    return new SimpleLinkerInput(artifact, category, disableWholeArchive, libraryIdentifier);
  }

  /** Creates input libraries for which we do not know what objects files it consists of. */
  public static Iterable<LibraryInput> opaqueLibrariesToLink(
      final ArtifactCategory category, Iterable<Artifact> input) {
    return Iterables.transform(input, artifact -> precompiledLibraryInput(artifact, category));
  }

  /** Creates a solib library symlink from the given artifact. */
  public static LibraryInput solibLibraryInput(
      Artifact solibSymlink, Artifact original, String libraryIdentifier) {
    return new SolibLibraryInput(solibSymlink, original, libraryIdentifier);
  }

  /** Creates an input library for which we do not know what objects files it consists of. */
  public static LibraryInput precompiledLibraryInput(Artifact artifact, ArtifactCategory category) {
    // This precondition check was in place and *most* of the tests passed with them; the only
    // exception is when you mention a generated .a file in the srcs of a cc_* rule.
    // It was very useful for proving that this actually works, though.
    // Preconditions.checkArgument(
    //     !(artifact.getGeneratingAction() instanceof CppLinkAction) ||
    //     !Link.ARCHIVE_LIBRARY_FILETYPES.contains(artifact.getFileType()));
    return new CompoundLibraryInput(
        artifact,
        category,
        CcLinkingOutputs.libraryIdentifierOf(artifact),
        /* objectFiles= */ null,
        /* ltoCompilationContext= */ null,
        /* sharedNonLtoBackends= */ null,
        /* allowArchiveTypeInAlwayslink= */ false,
        /* mustKeepDebug= */ false,
        /* disableWholeArchive= */ false);
  }

  /** Creates a library to link with the specified object files. */
  public static LibraryInput newInputLibrary(
      Artifact library,
      ArtifactCategory category,
      String libraryIdentifier,
      ImmutableCollection<Artifact> objectFiles,
      LtoCompilationContext ltoCompilationContext,
      ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends,
      boolean mustKeepDebug) {
    return newInputLibrary(
        library,
        category,
        libraryIdentifier,
        objectFiles,
        ltoCompilationContext,
        sharedNonLtoBackends,
        mustKeepDebug,
        /* disableWholeArchive= */ false);
  }

  /** Creates a library to link with the specified object files. */
  static LibraryInput newInputLibrary(
      Artifact library,
      ArtifactCategory category,
      String libraryIdentifier,
      ImmutableCollection<Artifact> objectFiles,
      LtoCompilationContext ltoCompilationContext,
      ImmutableMap<Artifact, LtoBackendArtifacts> sharedNonLtoBackends,
      boolean mustKeepDebug,
      boolean disableWholeArchive) {
    return new CompoundLibraryInput(
        library,
        category,
        libraryIdentifier,
        objectFiles,
        ltoCompilationContext,
        sharedNonLtoBackends,
        /* allowArchiveTypeInAlwayslink= */ true,
        mustKeepDebug,
        disableWholeArchive);
  }

  /** Returns the linker input artifacts from a collection of {@link LegacyLinkerInput} objects. */
  public static Iterable<Artifact> toLibraryArtifacts(
      Iterable<? extends LegacyLinkerInput> artifacts) {
    return Iterables.transform(artifacts, LegacyLinkerInput::getArtifact);
  }
}
