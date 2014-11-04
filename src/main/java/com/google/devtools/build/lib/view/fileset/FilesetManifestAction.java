// Copyright 2014 Google Inc. All rights reserved.
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

package com.google.devtools.build.lib.view.fileset;

import static java.nio.charset.StandardCharsets.ISO_8859_1;

import com.google.devtools.build.lib.actions.ActionOwner;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.ResourceSet;
import com.google.devtools.build.lib.actions.UserExecException;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.view.actions.AbstractFileWriteAction;

import java.io.BufferedWriter;
import java.io.IOException;
import java.io.OutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.util.Map;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * Action to create the input manifest for creating a Fileset tree.
 *
 * <p>This code has been adapted from {@code SourceSymlinkManifestAction}.
 *
 * <p>Unlike the SourceSymlinkManifestAction, this action does not know all of
 * its inputs at analysis time. In such cases, the given {@link
 * SymlinkTraversal} finds the inputs at execution time.
 *
 * <p>This deficiency would be mitigated by allowing recursive globs. Such a
 * feature would create complications of its own (for instance, what to do if
 * the glob crosses package boundaries).
 *
 * <p>Because the action does not know all of its inputs, it must be run on
 * every build.
 */
public class FilesetManifestAction extends AbstractFileWriteAction {

  private final SymlinkTraversal traversal;
  private final Artifact outputManifest;

  private final String FILESET_GUID = "0e0d0fe0-cf05-42e4-8c6e-c52664f35c49";

  /**
   * Creates a new FilesetManifestAction instance using ISO-LATIN-1 encoding to write
   * the manifest file.
   *
   * @param owner the action owner.
   * @param output the manifest file to write the symlinks to.
   */
  public FilesetManifestAction(ActionOwner owner, SymlinkTraversal traversal,
      NestedSet<Artifact> inputs, Artifact output) {
    super(owner, inputs, output, false);
    this.traversal = traversal;
    this.outputManifest = output;
  }

  private void checkForSpace(PathFragment file) throws UserExecException {
    if (file.toString().contains(" ")) {
      throw new UserExecException(
          String.format("Fileset '%s' includes a file with a space: '%s'",
                        getOwner().getLabel(), file));
    }
  }

  @Override
  public void writeOutputFile(OutputStream out, EventHandler eventHandler,
      Executor executor) throws IOException, InterruptedException, ExecException {
    // TODO(bazel-team): factor out common code from RunfilesManifestAction.
    Writer manifest = new BufferedWriter(new OutputStreamWriter(out, ISO_8859_1));
    FilesetLinks links = new FilesetLinks();
    FilesetActionContext context = executor.getContext(FilesetActionContext.class);
    try {
      ThreadPoolExecutor filesetPool = context.getFilesetPool();
      traversal.addSymlinks(eventHandler, links, filesetPool);
    } catch (BadSubpackageException e) {
      throw new UserExecException(""); // Error was already reported.
    } catch (DanglingSymlinkException e) {
      throw new EnvironmentalExecException("Found dangling symlink: " + e.getPath());
    }

    links.addLateDirectories();

    Map<PathFragment, String> data = links.getData();
    for (Map.Entry<PathFragment, PathFragment> line : links.getSymlinks().entrySet()) {
      PathFragment link = line.getKey();
      PathFragment target = line.getValue();
      checkForSpace(link);
      checkForSpace(target);

      if (!context.getWorkspaceName().isEmpty()) {
        manifest.append(context.getWorkspaceName() + "/");
      }

      manifest.append(link.getPathString());
      manifest.append(' ');

      manifest.append(line.getValue().getPathString());
      manifest.append('\n');
      manifest.append(data.get(link));
      manifest.append('\n');
    }
    manifest.flush();
  }

  @Override
  public String getMnemonic() { return "FilesetTraversal"; }

  @Override
  protected String getRawProgressMessage() {
    return "Traversing Fileset trees to write manifest " + outputManifest.prettyPrint();
  }

  @Override
  protected String computeKey() {
    Fingerprint fp = new Fingerprint();
    fp.addString(FILESET_GUID);
    traversal.fingerprint(fp);
    return fp.hexDigest();
  }

  @Override
  public boolean executeUnconditionally() {
    // Note: isVolatile must return true if executeUnconditionally can ever return true
    // for this instance.
    return traversal.executeUnconditionally();
  }

  @Override
  public boolean isVolatile() {
    return traversal.isVolatile();
  }
  
  // Can consume a lot of CPU, but several concurrent FilesetManifestAction instances will play
  // nicely together, as they consume a shared thread pool.
  private static final ResourceSet FILESET_MANIFEST_ACTION_RESOURCE_SET =
      new ResourceSet(/*memoryMb=*/100, /*cpuUsage=*/.2, /*ioUsage=*/0.1);

  @Override
  public ResourceSet estimateResourceConsumption(Executor executor) {
    return FILESET_MANIFEST_ACTION_RESOURCE_SET;
  }  
}
