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

package com.google.devtools.build.lib.sandbox;

import com.google.devtools.build.lib.exec.TreeDeleter;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;

/** Executes file system tree deletions synchronously. */
public class SynchronousTreeDeleter implements TreeDeleter {

  @Override
  public void deleteTree(Path path) throws IOException {
    path.deleteTree();
  }

  @Override
  public void deleteTreesBelow(Path path) throws IOException {
    path.deleteTreesBelow();
  }

  @Override
  public void shutdown() {}
}
