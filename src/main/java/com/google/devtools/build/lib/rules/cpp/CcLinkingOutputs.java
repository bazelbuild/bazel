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
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.rules.cpp.LinkerInputs.LibraryToLink;
import com.google.devtools.build.lib.skylarkbuildapi.cpp.CcLinkingOutputsApi;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import javax.annotation.Nullable;

/** A structured representation of the link outputs of a C++ rule. */
public class CcLinkingOutputs implements CcLinkingOutputsApi<Artifact> {

  public static final CcLinkingOutputs EMPTY = new Builder().build();

  @Nullable private final LibraryToLinkWrapper libraryToLink;

  private final ImmutableList<LtoBackendArtifacts> allLtoArtifacts;
  private final ImmutableList<Artifact> linkActionInputs;

  private CcLinkingOutputs(
      LibraryToLinkWrapper libraryToLink,
      ImmutableList<LtoBackendArtifacts> allLtoArtifacts,
      ImmutableList<Artifact> linkActionInputs) {
    this.libraryToLink = libraryToLink;
    this.allLtoArtifacts = allLtoArtifacts;
    this.linkActionInputs = linkActionInputs;
  }

  @Override
  @Nullable
  public LibraryToLinkWrapper getLibraryToLink() {
    return libraryToLink;
  }

  public ImmutableList<LtoBackendArtifacts> getAllLtoArtifacts() {
    return allLtoArtifacts;
  }

  public ImmutableList<Artifact> getLinkActionInputs() {
    return linkActionInputs;
  }

  public boolean isEmpty() {
    return libraryToLink == null;
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
    private LibraryToLinkWrapper libraryToLink;

    // TODO(plf): Return a list of debug artifacts instead of lto back end artifacts and in that
    // same list return the .pdb file for Windows.
    private final ImmutableList.Builder<LtoBackendArtifacts> allLtoArtifacts =
        ImmutableList.builder();
    private final ImmutableList.Builder<Artifact> linkActionInputs = ImmutableList.builder();

    public CcLinkingOutputs build() {
      return new CcLinkingOutputs(libraryToLink, allLtoArtifacts.build(), linkActionInputs.build());
    }

    public Builder setLibraryToLink(LibraryToLinkWrapper libraryToLink) {
      this.libraryToLink = libraryToLink;
      return this;
    }

    public Builder addAllLtoArtifacts(Iterable<LtoBackendArtifacts> allLtoArtifacts) {
      this.allLtoArtifacts.addAll(allLtoArtifacts);
      return this;
    }

    public Builder addLinkActionInputs(Iterable<Artifact> linkActionInputs) {
      this.linkActionInputs.addAll(linkActionInputs);
      return this;
    }
  }
}
