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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.collect.ImmutableSortedMap.toImmutableSortedMap;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
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
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.PrintWriter;
import java.util.Map.Entry;
import java.util.UUID;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** Creates a manifest file describing the repos and mappings relevant for a runfile tree. */
public final class RepoMappingManifestAction extends AbstractFileWriteAction {

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

  private final NestedSet<Package> transitivePackages;
  private final NestedSet<Artifact> runfilesArtifacts;
  private final String workspaceName;

  public RepoMappingManifestAction(
      ActionOwner owner,
      Artifact output,
      NestedSet<Package> transitivePackages,
      NestedSet<Artifact> runfilesArtifacts,
      String workspaceName) {
    super(owner, NestedSetBuilder.emptySet(Order.STABLE_ORDER), output, /*makeExecutable=*/ false);
    this.transitivePackages = transitivePackages;
    this.runfilesArtifacts = runfilesArtifacts;
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
    actionKeyContext.addNestedSetToFingerprint(fp, runfilesArtifacts);
    fp.addString(workspaceName);
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)
      throws InterruptedException, ExecException {
    return out -> {
      PrintWriter writer = new PrintWriter(out, /* autoFlush= */ false, ISO_8859_1);

      ImmutableSet<RepositoryName> reposContributingRunfiles =
          runfilesArtifacts.toList().stream()
              .filter(a -> a.getOwner() != null)
              .map(a -> a.getOwner().getRepository())
              .collect(toImmutableSet());
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
                  writeRepoMapping(writer, reposContributingRunfiles, repoName, mapping));
      writer.flush();
    };
  }

  private void writeRepoMapping(
      PrintWriter writer,
      ImmutableSet<RepositoryName> reposContributingRunfiles,
      RepositoryName repoName,
      RepositoryMapping repoMapping) {
    for (Entry<String, RepositoryName> mappingEntry :
        ImmutableSortedMap.copyOf(repoMapping.entries()).entrySet()) {
      if (mappingEntry.getKey().isEmpty()) {
        // The apparent repo name can only be empty for the main repo. We skip this line as
        // Rlocation paths can't reference an empty apparent name anyway.
        continue;
      }
      if (!reposContributingRunfiles.contains(mappingEntry.getValue())) {
        // We only write entries for repos that actually contribute runfiles.
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
