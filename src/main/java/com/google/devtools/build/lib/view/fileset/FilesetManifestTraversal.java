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

import com.google.common.base.Preconditions;
import com.google.common.base.Splitter;
import com.google.devtools.build.lib.events.ErrorEventListener;
import com.google.devtools.build.lib.pkgcache.PackageUpToDateChecker;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * This class does a Fileset traversal over the output tree of another
 * Fileset on which it depends. It uses the dependency's manifest instead
 * of actually traversing the filesystem for performance reasons.
 */
final class FilesetManifestTraversal implements SymlinkTraversal {

  private final Path manifest;
  private final PathFragment destDir;
  private final Collection<String> excludes;

  private static final String UUID = "78fc8035-039c-4519-ad3c-6c61d5928bd8";

  /**
   * Create the traversal over the manifest file.
   *
   * @param manifest The path to the other Fileset's manifest.
   * @param destDir The destdir for this traversal.
   * @param excludes The excludes for this traversal.
   */
  public FilesetManifestTraversal(Path manifest, PathFragment destDir,
                                  Collection<String> excludes) {
    this.manifest = manifest;
    this.destDir = destDir;
    this.excludes = excludes;
  }

  @Override
  public void addSymlinks(ErrorEventListener listener, FilesetLinks links,
      ThreadPoolExecutor filesetPool) throws IOException {
    // The manifest file is written in build-runfiles.cc.
    // See the documentation in that file for format specification.
    //
    // In the case of Filesets, the file has 2N lines, where N is the number
    // of symlinks. Each link is printed on two contiguous lines such as:
    //   <workspace>/symlink/location [/]symlink/target
    //   <opaque metadata>

    // ISO_8859 is used to write the manifest in {Runfiles,Fileset}ManifestAction.
    BufferedReader reader =
        new BufferedReader(new InputStreamReader(manifest.getInputStream(), ISO_8859_1));
    String line;

    try {
      while ((line = reader.readLine()) != null) {
        Preconditions.checkState(line.startsWith("google3/"));
        Iterator<String> tokens = Splitter.on(" ").split(line).iterator();

        Preconditions.checkState(tokens.hasNext(), line);
        String first = tokens.next();
        
        Preconditions.checkState(tokens.hasNext(), line);
        String second = tokens.next();
        
        Preconditions.checkState(!tokens.hasNext(), line);

        PathFragment src = new PathFragment(first.substring("google3/".length()));
        PathFragment target = new PathFragment(second);

        // Now read the metadata.
        line = reader.readLine();
        Preconditions.checkNotNull(line);

        if (!excludes.contains(src.getSegment(0))) {
          links.addLink(destDir.getRelative(src), target, line);
        }
      }
    } finally {
      reader.close();
    }
  }

  @Override
  public void fingerprint(Fingerprint fp) {
    fp.addString(UUID);
    fp.addPath(manifest);
    fp.addPath(destDir);
    fp.addStrings(excludes);
  }

  @Override
  public boolean executeUnconditionally(PackageUpToDateChecker upToDateChecker) {
    // Note: isVolatile must return true if executeUnconditionally can ever return true
    // for this instance.
    return false;
  }

  @Override
  public boolean isVolatile() {
    return false;
  }
}
