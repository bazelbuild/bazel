// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableSortedMap.toImmutableSortedMap;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;
import static java.util.Comparator.comparing;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSortedMap;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.CommandLineItem.MapFn;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.Map.Entry;
import java.util.UUID;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** Creates a manifest file describing the repos and mappings relevant for a runfile tree. */
public final class RepoMappingManifestAction extends AbstractFileWriteAction
    implements AbstractFileWriteAction.FileContentsProvider {

  private static final UUID MY_UUID = UUID.fromString("458e351c-4d30-433d-b927-da6cddd4737f");

  // Uses MapFn's args parameter just like Fingerprint#addString to compute a cacheable fingerprint
  // of just the repo name and mapping of a given Package.
  private static final MapFn<Package> REPO_AND_MAPPING_DIGEST_FN =
      (pkg, args) -> {
        args.accept(pkg.getPackageIdentifier().getRepository().getName());

        var mapping = pkg.getRepositoryMapping().entries();
        args.accept(String.valueOf(mapping.size()));
        mapping.forEach(
            (apparentName, canonicalName) -> {
              args.accept(apparentName);
              args.accept(canonicalName.getName());
            });
      };

  private static final MapFn<Artifact> OWNER_REPO_FN =
      (artifact, args) -> {
        args.accept(
            artifact.getOwner() != null ? artifact.getOwner().getRepository().getName() : "");
      };

  private static final MapFn<SymlinkEntry> FIRST_SEGMENT_FN =
      (symlink, args) -> args.accept(symlink.getPath().getSegment(0));

  private final NestedSet<Package> transitivePackages;
  private final NestedSet<Artifact> runfilesArtifacts;
  private final boolean hasRunfilesSymlinks;
  private final NestedSet<SymlinkEntry> runfilesRootSymlinks;
  private final String workspaceName;

  public RepoMappingManifestAction(
      ActionOwner owner,
      Artifact output,
      NestedSet<Package> transitivePackages,
      NestedSet<Artifact> runfilesArtifacts,
      NestedSet<SymlinkEntry> runfilesSymlinks,
      NestedSet<SymlinkEntry> runfilesRootSymlinks,
      String workspaceName) {
    super(
        owner, NestedSetBuilder.emptySet(Order.STABLE_ORDER), output, /* makeExecutable= */ false);
    this.transitivePackages = transitivePackages;
    this.runfilesArtifacts = runfilesArtifacts;
    this.hasRunfilesSymlinks = !runfilesSymlinks.isEmpty();
    this.runfilesRootSymlinks = runfilesRootSymlinks;
    this.workspaceName = workspaceName;
  }

  @Override
  public String getMnemonic() {
    return "RepoMappingManifest";
  }

  @Override
  protected String getRawProgressMessage() {
    return "Writing repo mapping manifest for " + getOwner().getLabel();
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp)
      throws CommandLineExpansionException, EvalException, InterruptedException {
    fp.addUUID(MY_UUID);
    actionKeyContext.addNestedSetToFingerprint(REPO_AND_MAPPING_DIGEST_FN, fp, transitivePackages);
    actionKeyContext.addNestedSetToFingerprint(OWNER_REPO_FN, fp, runfilesArtifacts);
    fp.addBoolean(hasRunfilesSymlinks);
    actionKeyContext.addNestedSetToFingerprint(FIRST_SEGMENT_FN, fp, runfilesRootSymlinks);
    fp.addString(workspaceName);
  }

  /**
   * Get the contents of a file internally using an in memory output stream.
   *
   * @return returns the file contents as a string.
   */
  @Override
  public String getFileContents(@Nullable EventHandler eventHandler) throws IOException {
    ByteArrayOutputStream stream = new ByteArrayOutputStream();
    newDeterministicWriter().writeOutputFile(stream);
    return stream.toString(UTF_8);
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    return newDeterministicWriter();
  }

  public DeterministicWriter newDeterministicWriter() {
    return out -> {
      PrintWriter writer = new PrintWriter(out, /* autoFlush= */ false, ISO_8859_1);

      var reposInRunfilePaths = ImmutableSet.<String>builder();

      // The runfiles paths of symlinks are always prefixed with the main workspace name, *not* the
      // name of the repository adding the symlink.
      if (hasRunfilesSymlinks) {
        reposInRunfilePaths.add(RepositoryName.MAIN.getName());
      }

      // Since root symlinks are the only way to stage a runfile at a specific path under the
      // current repository's runfiles directory, recognize canonical repository names that appear
      // as the first segment of their runfiles paths.
      for (SymlinkEntry symlink : runfilesRootSymlinks.toList()) {
        reposInRunfilePaths.add(symlink.getPath().getSegment(0));
      }

      for (Artifact artifact : runfilesArtifacts.toList()) {
        Label owner = artifact.getOwner();
        if (owner != null) {
          reposInRunfilePaths.add(owner.getRepository().getName());
        }
      }

      transitivePackages.toList().stream()
          .collect(
              toImmutableSortedMap(
                  comparing(RepositoryName::getName),
                  pkg -> pkg.getPackageIdentifier().getRepository(),
                  Package::getRepositoryMapping,
                  // All packages in a given repository have the same repository mapping, so the
                  // particular way of resolving duplicates does not matter.
                  (first, second) -> first))
          .forEach(
              (repoName, mapping) ->
                  writeRepoMapping(writer, reposInRunfilePaths.build(), repoName, mapping));
      writer.flush();
    };
  }

  private void writeRepoMapping(
      PrintWriter writer,
      ImmutableSet<String> reposInRunfilesPaths,
      RepositoryName repoName,
      RepositoryMapping repoMapping) {
    for (Entry<String, RepositoryName> mappingEntry :
        ImmutableSortedMap.copyOf(repoMapping.entries()).entrySet()) {
      if (mappingEntry.getKey().isEmpty()) {
        // The apparent repo name can only be empty for the main repo. We skip this line as
        // Rlocation paths can't reference an empty apparent name anyway.
        continue;
      }
      if (!reposInRunfilesPaths.contains(mappingEntry.getValue().getName())) {
        // We only write entries for repos whose canonical names appear in runfiles paths.
        continue;
      }
      // The canonical name of the main repo is the empty string, which is not a valid name for a
      // directory, so the "workspace name" is used the name of the directory under the runfiles
      // tree for it.
      String targetRepoDirectoryName =
          mappingEntry.getValue().isMain() ? workspaceName : mappingEntry.getValue().getName();
      writer.format(
          "%s,%s,%s\n", repoName.getName(), mappingEntry.getKey(), targetRepoDirectoryName);
    }
  }
}
