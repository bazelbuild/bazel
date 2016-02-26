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

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionExecutionContext;
import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.actions.AbstractFileWriteAction;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;

import javax.annotation.Nullable;

/**
 * Action to create a manifest of input files for processing by a subsequent
 * build step (e.g. runfiles symlinking or archive building).
 *
 * <p>The manifest's format is specifiable by {@link ManifestType}, in
 * accordance with the needs of the calling functionality.
 *
 * <p>Note that this action carefully avoids building the manifest content in
 * memory.
 */
public class SourceManifestAction extends AbstractFileWriteAction {

  private static final String GUID = "07459553-a3d0-4d37-9d78-18ed942470f4";

  /**
   * Interface for defining manifest formatting and reporting specifics.
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
    void writeEntry(Writer manifestWriter, PathFragment rootRelativePath,
        @Nullable Artifact symlink) throws IOException;

    /**
     * Fulfills {@link #ActionMetadata.getMnemonic()}
     */
    String getMnemonic();

    /**
     * Fulfills {@link #AbstractAction.getRawProgressMessage()}
     */
    String getRawProgressMessage();
  }

  /**
   * The strategy we use to write manifest entries.
   */
  private final ManifestWriter manifestWriter;

  /**
   * The runfiles for which to create the symlink tree.
   */
  private final Runfiles runfiles;

  /**
   * Creates a new AbstractSourceManifestAction instance using latin1 encoding
   * to write the manifest file and with a specified root path for manifest entries.
   *
   * @param manifestWriter the strategy to use to write manifest entries
   * @param owner the action owner
   * @param output the file to which to write the manifest
   * @param runfiles runfiles
   */
  private SourceManifestAction(ManifestWriter manifestWriter, ActionOwner owner, Artifact output,
      Runfiles runfiles) {
    super(owner, getDependencies(runfiles), output, false);
    this.manifestWriter = manifestWriter;
    this.runfiles = runfiles;
  }

  @VisibleForTesting
  public void writeOutputFile(OutputStream out, EventHandler eventHandler)
      throws IOException {
    writeFile(out, runfiles.getRunfilesInputs(eventHandler, getOwner().getLocation()));
  }

  @Override
  public DeterministicWriter newDeterministicWriter(ActionExecutionContext ctx)
      throws IOException {
    final Map<PathFragment, Artifact> runfilesInputs =
        runfiles.getRunfilesInputs(ctx.getExecutor().getEventHandler(), getOwner().getLocation());
    return new DeterministicWriter() {
      @Override
      public void writeOutputFile(OutputStream out) throws IOException {
        writeFile(out, runfilesInputs);
      }
    };
  }

  @Override
  public boolean isRemotable() {
    // There is little gain to remoting these, since they include absolute path names inline.
    return false;
  }

  /**
   * Returns the input dependencies for this action. Note we don't need to create the symlink
   * target Artifacts before we write the output manifest, so this Action does not have to
   * depend on them. The only necessary dependencies are pruning manifests, which must be read
   * to properly prune the tree.
   */
  private static Collection<Artifact> getDependencies(Runfiles runfiles) {
    ImmutableList.Builder<Artifact> builder = ImmutableList.builder();
    for (Runfiles.PruningManifest manifest : runfiles.getPruningManifests()) {
      builder.add(manifest.getManifestFile());
    }
    return builder.build();
  }

  /**
   * Sort the entries in both the normal and root manifests and write the output
   * file.
   *
   * @param out is the message stream to write errors to.
   * @param output The actual mapping of the output manifest.
   * @throws IOException
   */
  private void writeFile(OutputStream out, Map<PathFragment, Artifact> output) throws IOException {
    Writer manifestFile = new BufferedWriter(new OutputStreamWriter(out, ISO_8859_1));

    Comparator<Map.Entry<PathFragment, Artifact>> fragmentComparator =
          new Comparator<Map.Entry<PathFragment, Artifact>>() {
      @Override
      public int compare(Map.Entry<PathFragment, Artifact> path1,
                         Map.Entry<PathFragment, Artifact> path2) {
        return path1.getKey().compareTo(path2.getKey());
      }
    };

    List<Map.Entry<PathFragment, Artifact>> sortedManifest = new ArrayList<>(output.entrySet());
    Collections.sort(sortedManifest, fragmentComparator);

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
  protected String computeKey() {
    Fingerprint f = new Fingerprint();
    f.addString(GUID);
    Map<PathFragment, Artifact> symlinks = runfiles.getSymlinksAsMap();
    f.addInt(symlinks.size());
    for (Map.Entry<PathFragment, Artifact> symlink : symlinks.entrySet()) {
      f.addPath(symlink.getKey());
      f.addPath(symlink.getValue().getPath());
    }
    Map<PathFragment, Artifact> rootSymlinks = runfiles.getRootSymlinksAsMap();
    f.addInt(rootSymlinks.size());
    for (Map.Entry<PathFragment, Artifact> rootSymlink : rootSymlinks.entrySet()) {
      f.addPath(rootSymlink.getKey());
      f.addPath(rootSymlink.getValue().getPath());
    }

    for (Artifact artifact : runfiles.getArtifactsWithoutMiddlemen()) {
      f.addPath(artifact.getRootRelativePath());
      f.addPath(artifact.getPath());
    }
    return f.hexDigestAndReset();
  }

  /**
   * Supported manifest writing strategies.
   */
  public static enum ManifestType implements ManifestWriter {

    /**
     * Writes each line as:
     *
     * [rootRelativePath] [resolvingSymlink]
     *
     * <p>This strategy is suitable for creating an input manifest to a source view tree. Its
     * output is a valid input to {@link com.google.devtools.build.lib.analysis.SymlinkTreeAction}.
     */
    SOURCE_SYMLINKS {
      @Override
      public void writeEntry(Writer manifestWriter, PathFragment rootRelativePath, Artifact symlink)
          throws IOException {
        manifestWriter.append(rootRelativePath.getPathString());
        // This trailing whitespace is REQUIRED to process the single entry line correctly.
        manifestWriter.append(' ');
        if (symlink != null) {
          manifestWriter.append(symlink.getPath().getPathString());
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
    },

    /**
     * Writes each line as:
     *
     * [rootRelativePath]
     *
     * <p>This strategy is suitable for an input into a packaging system (notably .par) that
     * consumes a list of all source files but needs that list to be constant with respect to
     * how the user has their client laid out on local disk.
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
    }
  }

  /** Creates an action for the given runfiles. */
  public static SourceManifestAction forRunfiles(ManifestType manifestType, ActionOwner owner,
      Artifact output, Runfiles runfiles) {
    return new SourceManifestAction(manifestType, owner, output, runfiles);
  }

  /**
   * Builder class to construct {@link SourceManifestAction} instances.
   */
  public static final class Builder {
    private final ManifestWriter manifestWriter;
    private final ActionOwner owner;
    private final Artifact output;
    private final Runfiles.Builder runfilesBuilder;

    public Builder(String prefix, ManifestType manifestType, ActionOwner owner, Artifact output) {
      this.runfilesBuilder = new Runfiles.Builder(prefix);
      manifestWriter = manifestType;
      this.owner = owner;
      this.output = output;
    }

    @VisibleForTesting
    Builder(String prefix, ManifestWriter manifestWriter, ActionOwner owner, Artifact output) {
      this.runfilesBuilder = new Runfiles.Builder(prefix);
      this.manifestWriter = manifestWriter;
      this.owner = owner;
      this.output = output;
    }

    public SourceManifestAction build() {
      return new SourceManifestAction(manifestWriter, owner, output, runfilesBuilder.build());
    }

    /**
     * Adds a set of symlinks from the artifacts' root-relative paths to the
     * artifacts themselves.
     */
    public Builder addSymlinks(Iterable<Artifact> artifacts) {
      runfilesBuilder.addArtifacts(artifacts);
      return this;
    }

    /**
     * Adds a map of symlinks.
     */
    public Builder addSymlinks(Map<PathFragment, Artifact> symlinks) {
      runfilesBuilder.addSymlinks(symlinks);
      return this;
    }

    /**
     * Adds a single symlink.
     */
    public Builder addSymlink(PathFragment link, Artifact target) {
      runfilesBuilder.addSymlink(link, target);
      return this;
    }

    /**
     * <p>Adds a mapping of Artifacts to the directory above the normal symlink
     * forest base.
     */
    public Builder addRootSymlinks(Map<PathFragment, Artifact> rootSymlinks) {
      runfilesBuilder.addRootSymlinks(rootSymlinks);
      return this;
    }

    /**
     * Set the empty files supplier for the manifest, see {@link Runfiles.EmptyFilesSupplier}
     * for more details.
     */
    public Builder setEmptyFilesSupplier(Runfiles.EmptyFilesSupplier supplier) {
      runfilesBuilder.setEmptyFilesSupplier(supplier);
      return this;
    }

    /**
     * Adds a runfiles pruning manifest.
     */
    @VisibleForTesting
    Builder addPruningManifest(Runfiles.PruningManifest manifest) {
      runfilesBuilder.addPruningManifest(manifest);
      return this;
    }
  }
}
