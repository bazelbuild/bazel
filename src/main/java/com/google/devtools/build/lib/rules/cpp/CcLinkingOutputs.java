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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.starlarkbuildapi.cpp.CcLinkingOutputsApi;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkThread;

/** A structured representation of the link outputs of a C++ rule. */
public class CcLinkingOutputs implements CcLinkingOutputsApi<Artifact, LtoBackendArtifacts> {

  public static final CcLinkingOutputs EMPTY = builder().build();

  @Nullable private final LibraryToLink libraryToLink;
  @Nullable private final Artifact executable;

  private final ImmutableList<LtoBackendArtifacts> allLtoArtifacts;

  private CcLinkingOutputs(
      LibraryToLink libraryToLink,
      Artifact executable,
      ImmutableList<LtoBackendArtifacts> allLtoArtifacts) {
    this.libraryToLink = libraryToLink;
    this.executable = executable;
    this.allLtoArtifacts = allLtoArtifacts;
  }

  @Override
  @Nullable
  public LibraryToLink getLibraryToLink() {
    return libraryToLink;
  }

  @Override
  @Nullable
  public Artifact getExecutable() {
    return executable;
  }

  public ImmutableList<LtoBackendArtifacts> getAllLtoArtifacts() {
    return allLtoArtifacts;
  }

  @Override
  public Sequence<LtoBackendArtifacts> getAllLtoArtifactsForStarlark(StarlarkThread thread)
      throws EvalException {
    CcModule.checkPrivateStarlarkificationAllowlist(thread);
    return StarlarkList.immutableCopyOf(getAllLtoArtifacts());
  }

  public boolean isEmpty() {
    return libraryToLink == null;
  }

  private static final ImmutableList<String> PIC_SUFFIXES =
      ImmutableList.of(".pic.a", ".nopic.a", ".pic.lo");

  /**
   * Returns the library identifier of an artifact: a string that is different for different
   * libraries, but is the same for the shared, static and pic versions of the same library.
   */
  public static String libraryIdentifierOf(Artifact libraryArtifact) {
    String name = libraryArtifact.getRootRelativePath().getPathString();
    for (String picSuffix : PIC_SUFFIXES) {
      if (name.endsWith(picSuffix)) {
        return name.substring(0, name.length() - picSuffix.length());
      }
    }
    return FileSystemUtils.removeExtension(name);
  }

  public static Builder builder() {
    return new Builder();
  }

  /** Builder for {@link CcLinkingOutputs} */
  public static final class Builder {
    private LibraryToLink libraryToLink;
    private Artifact executable;

    private Builder() {
      // private to avoid class initialization deadlock between this class and its outer class
    }

    // TODO(plf): Return a list of debug artifacts instead of lto back end artifacts and in that
    // same list return the .pdb file for Windows.
    private final ImmutableList.Builder<LtoBackendArtifacts> allLtoArtifacts =
        ImmutableList.builder();

    public CcLinkingOutputs build() {
      return new CcLinkingOutputs(libraryToLink, executable, allLtoArtifacts.build());
    }

    @CanIgnoreReturnValue
    public Builder setLibraryToLink(LibraryToLink libraryToLink) {
      this.libraryToLink = libraryToLink;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder setExecutable(Artifact executable) {
      this.executable = executable;
      return this;
    }

    @CanIgnoreReturnValue
    public Builder addAllLtoArtifacts(Iterable<LtoBackendArtifacts> allLtoArtifacts) {
      this.allLtoArtifacts.addAll(allLtoArtifacts);
      return this;
    }
  }
}
