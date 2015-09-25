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

package com.google.devtools.build.lib.rules.fileset;

import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Fingerprint;

import java.io.IOException;
import java.util.concurrent.ThreadPoolExecutor;

/**
 * An interface which contains a method to compute a symlink mapping.
 */
public interface SymlinkTraversal {

  /**
   * Adds symlinks to the given FilesetLinks.
   *
   * @throws IOException if a filesystem operation fails.
   * @throws InterruptedException if the traversal is interrupted.
   */
  void addSymlinks(EventHandler eventHandler, FilesetLinks links, ThreadPoolExecutor filesetPool)
      throws IOException, InterruptedException;

  /**
   * Add the traversal's fingerprint to the given Fingerprint.
   * @param fp the Fingerprint to combine.
   */
  void fingerprint(Fingerprint fp);

  /**
   * @return true iff this traversal must be executed unconditionally.
   */
  boolean executeUnconditionally();

  /**
   * Returns true if it's ever possible that {@link #executeUnconditionally}
   * could evaluate to true during the lifetime of this instance, false
   * otherwise.
   */
  boolean isVolatile();
}
