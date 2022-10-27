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

import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.util.Comparator.comparing;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.Artifact.ArtifactExpander;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.PrintWriter;
import java.util.List;
import java.util.UUID;
import javax.annotation.Nullable;
import net.starlark.java.eval.EvalException;

/** Creates a manifest file describing the repos and mappings relevant for a runfile tree. */
public class RepoMappingManifestAction extends AbstractFileWriteAction {
  private static final UUID MY_UUID = UUID.fromString("458e351c-4d30-433d-b927-da6cddd4737f");

  private final ImmutableList<Entry> entries;
  private final String workspaceName;

  /** An entry in the repo mapping manifest file. */
  @AutoValue
  public abstract static class Entry {
    public static Entry of(
        RepositoryName sourceRepo, String targetRepoApparentName, RepositoryName targetRepo) {
      return new AutoValue_RepoMappingManifestAction_Entry(
          sourceRepo, targetRepoApparentName, targetRepo);
    }

    public abstract RepositoryName sourceRepo();

    public abstract String targetRepoApparentName();

    public abstract RepositoryName targetRepo();
  }

  public RepoMappingManifestAction(
      ActionOwner owner, Artifact output, List<Entry> entries, String workspaceName) {
    super(owner, NestedSetBuilder.emptySet(Order.STABLE_ORDER), output, /*makeExecutable=*/ false);
    this.entries =
        ImmutableList.sortedCopyOf(
            comparing((Entry e) -> e.sourceRepo().getName())
                .thenComparing(Entry::targetRepoApparentName),
            entries);
    this.workspaceName = workspaceName;
  }

  @Override
  public String getMnemonic() {
    return "RepoMappingManifest";
  }

  @Override
  protected String getRawProgressMessage() {
    return "writing repo mapping manifest for " + getOwner().getLabel();
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable ArtifactExpander artifactExpander,
      Fingerprint fp)
      throws CommandLineExpansionException, EvalException, InterruptedException {
    fp.addUUID(MY_UUID);
    fp.addString(workspaceName);
    for (Entry entry : entries) {
      fp.addString(entry.sourceRepo().getName());
      fp.addString(entry.targetRepoApparentName());
      fp.addString(entry.targetRepo().getName());
    }
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)
      throws InterruptedException, ExecException {
    return out -> {
      PrintWriter writer = new PrintWriter(out, /*autoFlush=*/ false, ISO_8859_1);
      for (Entry entry : entries) {
        if (entry.targetRepoApparentName().isEmpty()) {
          // The apparent repo name can only be empty for the main repo. We skip this line as
          // Rlocation paths can't reference an empty apparent name anyway.
          continue;
        }
        // The canonical name of the main repo is the empty string, which is not a valid name for a
        // directory, so the "workspace name" is used the name of the directory under the runfiles
        // tree for it.
        String targetRepoDirectoryName =
            entry.targetRepo().isMain() ? workspaceName : entry.targetRepo().getName();
        writer.format(
            "%s,%s,%s\n",
            entry.sourceRepo().getName(), entry.targetRepoApparentName(), targetRepoDirectoryName);
      }
      writer.flush();
    };
  }
}
