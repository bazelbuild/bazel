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
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.cpp.Link.LinkingMode;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingOutputsApi;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.LibraryToLinkApi;
import com.google.devtools.build.lib.syntax.SkylarkList;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;

/** A structured representation of the link outputs of a C++ rule. */
public class CcLinkingOutputs implements CcLinkingOutputsApi {

  public static final CcLinkingOutputs EMPTY = new Builder().build();

  private final ImmutableList<LibraryToLink> staticLibraries;

  private final ImmutableList<LibraryToLink> picStaticLibraries;

  /**
   * Holds interface dynamic libraries if the toolchain supports them, full dynamic libraries
   * otherwise.
   *
   * <p>These are what dependants want to link against. We put these as inputs to the C++ link
   * action.
   */
  private final ImmutableList<LibraryToLink> dynamicLibrariesForLinking;

  /**
   * Holds dynamic libraries even when the toolchain supports interface libraries.
   *
   * <p>These are what binaries load at runtime. We put these into runfiles.
   */
  private final ImmutableList<LibraryToLink> dynamicLibrariesForRuntime;

  private final ImmutableList<LtoBackendArtifacts> allLtoArtifacts;

  private CcLinkingOutputs(
      ImmutableList<LibraryToLink> staticLibraries,
      ImmutableList<LibraryToLink> picStaticLibraries,
      ImmutableList<LibraryToLink> dynamicLibrariesForLinking,
      ImmutableList<LibraryToLink> dynamicLibrariesForRuntime,
      ImmutableList<LtoBackendArtifacts> allLtoArtifacts) {
    this.staticLibraries = staticLibraries;
    this.picStaticLibraries = picStaticLibraries;
    this.dynamicLibrariesForLinking = dynamicLibrariesForLinking;
    this.dynamicLibrariesForRuntime = dynamicLibrariesForRuntime;
    this.allLtoArtifacts = allLtoArtifacts;
  }

  @Override
  public SkylarkList<LibraryToLinkApi> getSkylarkStaticLibraries() {
    return SkylarkList.createImmutable(staticLibraries);
  }

  public ImmutableList<LibraryToLink> getStaticLibraries() {
    return staticLibraries;
  }

  @Override
  public SkylarkList<LibraryToLinkApi> getSkylarkPicStaticLibraries() {
    return SkylarkList.createImmutable(picStaticLibraries);
  }

  public ImmutableList<LibraryToLink> getPicStaticLibraries() {
    return picStaticLibraries;
  }

  @Override
  public SkylarkList<LibraryToLinkApi> getSkylarkDynamicLibrariesForLinking() {
    return SkylarkList.createImmutable(dynamicLibrariesForLinking);
  }

  public ImmutableList<LibraryToLink> getDynamicLibrariesForLinking() {
    return dynamicLibrariesForLinking;
  }

  public ImmutableList<LibraryToLink> getDynamicLibrariesForRuntime() {
    return dynamicLibrariesForRuntime;
  }

  public ImmutableList<LtoBackendArtifacts> getAllLtoArtifacts() {
    return allLtoArtifacts;
  }

  public boolean isEmpty() {
    return staticLibraries.isEmpty()
        && picStaticLibraries.isEmpty()
        && dynamicLibrariesForLinking.isEmpty()
        && dynamicLibrariesForRuntime.isEmpty();
  }

  /**
   * Returns a map from library identifiers to sets of LibraryToLink from this CcLinkingOutputs
   * which share that library identifier.
   */
  public ImmutableSetMultimap<String, LibraryToLink> getLibrariesByIdentifier() {
    return getLibrariesByIdentifier(
        Iterables.concat(
            staticLibraries,
            picStaticLibraries,
            dynamicLibrariesForLinking,
            dynamicLibrariesForRuntime));
  }

  /**
   * Gathers up a map from library identifiers to sets of LibraryToLink which share that library
   * identifier.
   */
  public static ImmutableSetMultimap<String, LibraryToLink> getLibrariesByIdentifier(
      Iterable<LibraryToLink> inputs) {
    ImmutableSetMultimap.Builder<String, LibraryToLink> result =
        new ImmutableSetMultimap.Builder<>();
    for (LibraryToLink library : inputs) {
      Preconditions.checkNotNull(library.getLibraryIdentifier());
      result.put(library.getLibraryIdentifier(), library);
    }
    return result.build();
  }

  /**
   * Returns the shared libraries that are linked against and therefore also need to be in the
   * runfiles.
   */
  public Iterable<Artifact> getLibrariesForRunfiles(boolean linkingStatically) {
    List<LibraryToLink> libraries =
        getPreferredLibraries(linkingStatically, /*preferPic*/ false, true);
    return PrecompiledFiles.getSharedLibrariesFrom(LinkerInputs.toLibraryArtifacts(libraries));
  }

  /**
   * Add the ".a", ".pic.a" and/or ".so" files in appropriate order of preference depending on the
   * link preferences.
   *
   * <p>This method tries to simulate a search path for adding static and dynamic libraries,
   * allowing either to be preferred over the other depending on the link {@link LinkingMode}.
   *
   * <p>TODO(bazel-team): (2009) we should preserve the relative ordering of first and second choice
   * libraries. E.g. if srcs=['foo.a','bar.so','baz.a'] then we should link them in the same order.
   * Currently we link entries from the first choice list before those from the second choice list,
   * i.e. in the order {@code ['bar.so', 'foo.a', 'baz.a']}.
   *
   * @param linkingStatically whether to prefer static over dynamic libraries. Should be <code>true
   *     </code> for binaries that are linked in fully static or mostly static mode.
   * @param preferPic whether to prefer pic over non pic libraries (usually used when linking
   *     shared)
   */
  public List<LibraryToLink> getPreferredLibraries(boolean linkingStatically, boolean preferPic) {
    return getPreferredLibraries(linkingStatically, preferPic, false);
  }

  /**
   * Add the ".a", ".pic.a" and/or ".so" files in appropriate order of
   * preference depending on the link preferences.
   */
  private List<LibraryToLink> getPreferredLibraries(boolean linkingStatically, boolean preferPic,
      boolean forRunfiles) {
    List<LibraryToLink> candidates = new ArrayList<>();
    // It's important that this code keeps the invariant that preferPic has no effect on the output
    // of .so libraries. That is, the resulting list should contain the same .so files in the same
    // order.
    if (linkingStatically) { // Prefer the static libraries.
      if  (preferPic) {
        // First choice is the PIC static libraries.
        // Second choice is the other static libraries (may cause link error if they're not PIC,
        // but I think this is preferable to linking dynamically when you asked for statically).
        candidates.addAll(picStaticLibraries);
        candidates.addAll(staticLibraries);
      } else {
        // First choice is the non-pic static libraries (best performance);
        // second choice is the staticPicLibraries (at least they're static;
        // we can live with the extra overhead of PIC).
        candidates.addAll(staticLibraries);
        candidates.addAll(picStaticLibraries);
      }
      candidates.addAll(forRunfiles ? dynamicLibrariesForRuntime : dynamicLibrariesForLinking);
    } else {
      // First choice is the dynamic libraries.
      candidates.addAll(forRunfiles ? dynamicLibrariesForRuntime : dynamicLibrariesForLinking);
      if (preferPic) {
        // Second choice is the staticPicLibraries (at least they're PIC, so we won't get a
        // link error).
        candidates.addAll(picStaticLibraries);
        candidates.addAll(staticLibraries);
      } else {
        candidates.addAll(staticLibraries);
        candidates.addAll(picStaticLibraries);
      }
    }
    return filterCandidates(candidates);
  }

  /**
   * Helper method to filter the candidates by removing equivalent library
   * entries from the list of candidates.
   *
   * @param candidates the library candidates to filter
   * @return the list of libraries with equivalent duplicate libraries removed.
   */
  private List<LibraryToLink> filterCandidates(List<LibraryToLink> candidates) {
    List<LibraryToLink> libraries = new ArrayList<>();
    Set<String> identifiers = new HashSet<>();
    for (LibraryToLink library : candidates) {
      Preconditions.checkNotNull(library.getLibraryIdentifier());
      if (identifiers.add(library.getLibraryIdentifier())) {
        libraries.add(library);
      }
    }
    return libraries;
  }

  /**
   * Returns the library identifier of an artifact: a string that is different for different
   * libraries, but is the same for the shared, static and pic versions of the same library.
   */
  public static String libraryIdentifierOf(Artifact libraryArtifact) {
    String name = libraryArtifact.getRootRelativePath().getPathString();
    String basename = FileSystemUtils.removeExtension(name);
    // Need to special-case file types with double extension.
    return name.endsWith(".pic.a")
        ? FileSystemUtils.removeExtension(basename)
        : name.endsWith(".nopic.a")
        ? FileSystemUtils.removeExtension(basename)
        : name.endsWith(".pic.lo")
        ? FileSystemUtils.removeExtension(basename)
        : basename;
  }

  public static Builder builder() {
    return new Builder();
  }

  public static final class Builder {
    private final Set<LibraryToLink> staticLibraries = new LinkedHashSet<>();
    private final Set<LibraryToLink> picStaticLibraries = new LinkedHashSet<>();
    private final Set<LibraryToLink> dynamicLibrariesForLinking = new LinkedHashSet<>();
    private final Set<LibraryToLink> dynamicLibrariesForRuntime = new LinkedHashSet<>();
    // TODO(plf): Return a list of debug artifacts instead of lto back end artifacts and in that
    // same list return the .pdb file for Windows.
    private final ImmutableList.Builder<LtoBackendArtifacts> allLtoArtifacts =
        ImmutableList.builder();

    public CcLinkingOutputs build() {
      return new CcLinkingOutputs(
          ImmutableList.copyOf(staticLibraries),
          ImmutableList.copyOf(picStaticLibraries),
          ImmutableList.copyOf(dynamicLibrariesForLinking),
          ImmutableList.copyOf(dynamicLibrariesForRuntime),
          allLtoArtifacts.build());
    }

    public Builder merge(CcLinkingOutputs outputs) {
      staticLibraries.addAll(outputs.getStaticLibraries());
      picStaticLibraries.addAll(outputs.getPicStaticLibraries());
      dynamicLibrariesForLinking.addAll(outputs.getDynamicLibrariesForLinking());
      dynamicLibrariesForRuntime.addAll(outputs.getDynamicLibrariesForRuntime());
      allLtoArtifacts.addAll(outputs.getAllLtoArtifacts());
      return this;
    }

    public Builder addStaticLibrary(LibraryToLink library) {
      staticLibraries.add(library);
      return this;
    }

    public Builder addStaticLibraries(Iterable<LibraryToLink> libraries) {
      Iterables.addAll(staticLibraries, libraries);
      return this;
    }

    public Builder addPicStaticLibrary(LibraryToLink library) {
      picStaticLibraries.add(library);
      return this;
    }

    public Builder addPicStaticLibraries(Iterable<LibraryToLink> libraries) {
      Iterables.addAll(picStaticLibraries, libraries);
      return this;
    }

    public Builder addDynamicLibraryForLinking(LibraryToLink library) {
      dynamicLibrariesForLinking.add(library);
      return this;
    }

    public Builder addDynamicLibraries(Iterable<LibraryToLink> libraries) {
      Iterables.addAll(dynamicLibrariesForLinking, libraries);
      return this;
    }

    public Builder addDynamicLibraryForRuntime(LibraryToLink library) {
      dynamicLibrariesForRuntime.add(library);
      return this;
    }

    public Builder addDynamicLibrariesForRuntime(Iterable<LibraryToLink> libraries) {
      Iterables.addAll(dynamicLibrariesForRuntime, libraries);
      return this;
    }

    public Builder addAllLtoArtifacts(Iterable<LtoBackendArtifacts> allLtoArtifacts) {
      this.allLtoArtifacts.addAll(allLtoArtifacts);
      return this;
    }
  }
}
