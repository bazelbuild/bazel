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
package com.google.devtools.build.lib.vfs;

import java.io.IOException;
import java.util.List;

/**
 * An interface for doing a batch of stat() calls.
 */
public interface BatchStat {

  /**
   *
   * @param includeDigest whether to include a file digest in the return values.
   * @param includeLinks whether to include a symlink stat in the return values.
   * @param paths The input paths to stat(), relative to the exec root.
   * @return an array list of FileStatusWithDigest in the same order as the input. May
   *         contain null values.
   * @throws IOException on unexpected failure.
   * @throws InterruptedException on interrupt.
   */
  public List<FileStatusWithDigest> batchStat(boolean includeDigest,
                                              boolean includeLinks,
                                              Iterable<PathFragment> paths)
      throws IOException, InterruptedException;
}
