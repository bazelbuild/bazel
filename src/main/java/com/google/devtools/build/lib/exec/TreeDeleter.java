// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.exec;

import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/** Interface to execute tree deletions under different scheduling policies. */
public interface TreeDeleter {

  /**
   * Deletes a tree.
   *
   * <p>If the given path does not exist, does nothing.
   *
   * <p>Note that depending on the scheduling policy implemented by this tree deleter, errors may
   * not be reported even if they happen. For example, if deletions are asynchronous, there is no
   * way to capture their errors.
   *
   * <p>For asynchronous deleters the directory is moved before being deleted.
   *
   * @param path the tree to be deleted
   * @throws IOException if an error occurs while deleting the tree
   */
  void deleteTree(Path path) throws IOException;

  /** Shuts down the tree deleter and cleans up pending operations, if any. */
  void shutdown();
}
