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
package com.google.devtools.build.lib.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static java.nio.charset.StandardCharsets.ISO_8859_1;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.AbstractAction;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionKeyContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.analysis.actions.DeterministicWriter;
import com.google.devtools.build.lib.analysis.starlark.UnresolvedSymlinkAction;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.concurrent.ThreadSafety.Immutable;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.BufferedWriter;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Creates a manifest file describing a symlink tree.
 *
 * <p>In addition to symlink trees (whose manifests are a tree position -> exec path map), this
 * action can also create manifest consisting of just exec paths for historical reasons.
 *
 * <p>This action carefully avoids building the manifest content in memory because it can be large.
 */
@Immutable // if all ManifestWriter implementations are immutable
public final class SourceManifestAction extends AbstractFileWriteAction
    implements AbstractFileWriteAction.FileContentsProvider {

  private static final String GUID = "07459553-a3d0-4d37-9d78-18ed942470f4";

  private static final Comparator<Map.Entry<PathFragment, Artifact>> ENTRY_COMPARATOR =
      (path1, path2) -> path1.getKey().getPathString().compareTo(path2.getKey().getPathString());
  private final Artifact repoMappingManifest;
  /**
   * Interface for defining manifest formatting and reporting specifics. Implementations must be
   * immutable.
   */
  @VisibleForTesting
  interface ManifestWriter {

    /**
     * Writes a single line of manifest output.
     *
     * @param manifestWriter the output stream
     * @param rootRelativePath path of an entry relative to the manifest's root
     * @param symlink (optional) symlink that resolves the above path
     */
    void writeEntry(
        Writer manifestWriter, PathFragment rootRelativePath, @Nullable Artifact symlink)
        throws IOException;

    /** Fulfills {@link com.google.devtools.build.lib.actions.AbstractAction#getMnemonic()} */
    String getMnemonic();

    /**
     * Fulfills {@link com.google.devtools.build.lib.actions.AbstractAction#getRawProgressMessage()}
     */
    String getRawProgressMessage();

    /**
     * Fulfills {@link AbstractFileWriteAction#isRemotable()}.
     *
     * @return
     */
    boolean isRemotable();
  }

  /** The strategy we use to write manifest entries. */
  private final ManifestWriter manifestWriter;

  /** The runfiles for which to create the symlink tree. */
  private final Runfiles runfiles;

  private final boolean remotableSourceManifestActions;

  private NestedSet<Artifact> symlinkArtifacts = null;

  /**
   * Creates a new AbstractSourceManifestAction instance using latin1 encoding to write the manifest
   * file and with a specified root path for manifest entries.
   *
   * @param manifestWriter the strategy to use to write manifest entries
   * @param owner the action owner
   * @param primaryOutput the file to which to write the manifest
   * @param runfiles runfiles
   */
  @VisibleForTesting
  SourceManifestAction(
      ManifestWriter manifestWriter, ActionOwner owner, Artifact primaryOutput, Runfiles runfiles) {
    this(manifestWriter, owner, primaryOutput, runfiles, null, false);
  }

  /**
   * Creates a new AbstractSourceManifestAction instance using latin1 encoding to write the manifest
   * file and with a specified root path for manifest entries.
   *
   * @param manifestWriter the strategy to use to write manifest entries
   * @param owner the action owner
   * @param primaryOutput the file to which to write the manifest
   * @param runfiles runfiles
   * @param repoMappingManifest the repository mapping manifest for runfiles
   */
  public SourceManifestAction(
      ManifestWriter manifestWriter,
      ActionOwner owner,
      Artifact primaryOutput,
      Runfiles runfiles,
      @Nullable Artifact repoMappingManifest,
      boolean remotableSourceManifestActions) {
    // The real set of inputs is computed in #getInputs().
    super(owner, NestedSetBuilder.emptySet(Order.STABLE_ORDER), primaryOutput, false);
    this.manifestWriter = manifestWriter;
    this.runfiles = runfiles;
    this.repoMappingManifest = repoMappingManifest;
    this.remotableSourceManifestActions = remotableSourceManifestActions;
  }

  /**
   * The manifest entry for a symlink artifact should contain the target of the symlink rather than
   * its exec path. Reading the symlink target requires that the symlink artifact is declared as an
   * input of this action. Since declaring all runfiles as inputs of the manifest action would
   * unnecessarily delay its execution, this action exceptionally overrides {@link
   * AbstractAction#getInputs()} and filters out the non-symlink runfiles by flattening the nested
   * set of runfiles. Benchmarks confirmed that this does not regress performance.
   *
   * <p>Alternatives considered:
   *
   * <ul>
   *   <li>Having users separate normal artifacts from symlink artifacts during analysis: Makes it
   *       impossible to pass symlink artifacts to rules that aren't aware of them and requires the
   *       use of custom providers to pass symlinks to stage as inputs to actions.
   *   <li>Reaching into {@link ActionExecutionContext} to look up the generating action of symlink
   *       artifacts and retrieving the target from {@link UnresolvedSymlinkAction}: This would not
   *       work for symlinks whose target is determined in the execution phase.
   *   <li>Input discovery: Complex and error-prone in general and conceptually not necessary here -
   *       we already know what the inputs will be during analysis, we just want to delay the
   *       required computations.
   * </ul>
   */
  @Override
  public synchronized NestedSet<Artifact> getInputs() {
    if (symlinkArtifacts == null) {
      ImmutableList<Artifact> symlinks =
          runfiles.getArtifacts().toList().stream()
              .filter(Artifact::isSymlink)
              .collect(toImmutableList());
      symlinkArtifacts = NestedSetBuilder.wrap(Order.STABLE_ORDER, symlinks);
    }
    return symlinkArtifacts;
  }

  @VisibleForTesting
  public void writeOutputFile(OutputStream out, @Nullable EventHandler eventHandler)
      throws IOException {
    writeFile(
        out,
        runfiles.getRunfilesInputs(eventHandler, getOwner().getLocation(), repoMappingManifest));
  }

  /**
   * Get the contents of a file internally using an in memory output stream.
   *
   * @return returns the file contents as a string.
   */
  @Override
  public String getFileContents(@Nullable EventHandler eventHandler) throws IOException {
    ByteArrayOutputStream stream = new ByteArrayOutputStream();
    writeOutputFile(stream, eventHandler);
    return stream.toString(UTF_8);
  }

  @Override
  public String getStarlarkContent() throws IOException {
    return getFileContents(null);
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx) {
    final Map<PathFragment, Artifact> runfilesInputs =
        runfiles.getRunfilesInputs(
            ctx.getEventHandler(), getOwner().getLocation(), repoMappingManifest);
    return out -> writeFile(out, runfilesInputs);
  }

  @Override
  public boolean isRemotable() {
    return remotableSourceManifestActions || manifestWriter.isRemotable();
  }

  /**
   * Sort the entries in both the normal and root manifests and write the output file.
   *
   * @param out is the message stream to write errors to.
   * @param output The actual mapping of the output manifest.
   * @throws IOException
   */
  private void writeFile(OutputStream out, Map<PathFragment, Artifact> output) throws IOException {
    Writer manifestFile = new BufferedWriter(new OutputStreamWriter(out, ISO_8859_1));
    List<Map.Entry<PathFragment, Artifact>> sortedManifest = new ArrayList<>(output.entrySet());
    sortedManifest.sort(ENTRY_COMPARATOR);
    for (Map.Entry<PathFragment, Artifact> line : sortedManifest) {
      manifestWriter.writeEntry(manifestFile, line.getKey(), line.getValue());
    }

    manifestFile.flush();
  }

  @Override
  public String getMnemonic() {
    return manifestWriter.getMnemonic();
  }

  @Override
  protected String getRawProgressMessage() {
    return manifestWriter.getRawProgressMessage() + " for " + getOwner().getLabel();
  }

  @Override
  protected void computeKey(
      ActionKeyContext actionKeyContext,
      @Nullable Artifact.ArtifactExpander artifactExpander,
      Fingerprint fp) {
    fp.addString(GUID);
    fp.addBoolean(remotableSourceManifestActions);
    runfiles.fingerprint(actionKeyContext, fp);
    fp.addBoolean(repoMappingManifest != null);
    if (repoMappingManifest != null) {
      fp.addPath(repoMappingManifest.getExecPath());
    }
  }

  @Override
  public String describeKey() {
    return String.format(
        "GUID: %s\nremotableSourceManifestActions: %s\nrunfiles: %s\n",
        GUID, remotableSourceManifestActions, runfiles.describeFingerprint());
  }

  /** Supported manifest writing strategies. */
  public enum ManifestType implements ManifestWriter {

    /**
     * Writes each line as:
     *
     * <p>[rootRelativePath] [resolvingSymlink]
     *
     * <p>This strategy is suitable for creating an input manifest to a source view tree. Its output
     * is a valid input to {@link com.google.devtools.build.lib.analysis.actions.SymlinkTreeAction}.
     */
    SOURCE_SYMLINKS {
      @Override
      public void writeEntry(Writer manifestWriter, PathFragment rootRelativePath, Artifact symlink)
          throws IOException {
        manifestWriter.append(rootRelativePath.getPathString());
        // This trailing whitespace is REQUIRED to process the single entry line correctly.
        manifestWriter.append(' ');
        if (symlink != null) {
          if (symlink.isSymlink()) {
            manifestWriter.append(symlink.getPath().readSymbolicLink().getPathString());
          } else {
            manifestWriter.append(symlink.getPath().getPathString());
          }
        }
        manifestWriter.append('\n');
      }

      @Override
      public String getMnemonic() {
        return "SourceSymlinkManifest";
      }

      @Override
      public String getRawProgressMessage() {
        return "Creating source manifest";
      }

      @Override
      public boolean isRemotable() {
        // There is little gain to remoting these, since they include absolute path names inline.
        return false;
      }
    },

    /**
     * Writes each line as:
     *
     * <p>[rootRelativePath]
     *
     * <p>This strategy is suitable for an input into a packaging system (notably .par) that
     * consumes a list of all source files but needs that list to be constant with respect to how
     * the user has their client laid out on local disk.
     */
    SOURCES_ONLY {
      @Override
      public void writeEntry(Writer manifestWriter, PathFragment rootRelativePath, Artifact symlink)
          throws IOException {
        manifestWriter.append(rootRelativePath.getPathString());
        manifestWriter.append('\n');
        manifestWriter.flush();
      }

      @Override
      public String getMnemonic() {
        return "PackagingSourcesManifest";
      }

      @Override
      public String getRawProgressMessage() {
        return "Creating file sources list";
      }

      @Override
      public boolean isRemotable() {
        // Source-only symlink manifest has root-relative paths and does not include absolute paths.
        return true;
      }
    }
  }
}
