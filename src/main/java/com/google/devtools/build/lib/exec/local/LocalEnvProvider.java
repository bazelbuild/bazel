// Copyright 2017 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.exec.local;

import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Map;

/**
 * Allows just-in-time rewriting of the environment used for local actions. Do not use! This class
 * probably should not exist, but is currently necessary for our local MacOS support.
 */
public interface LocalEnvProvider {

  public static final LocalEnvProvider UNMODIFIED =
      new LocalEnvProvider() {
        @Override
        public Map<String, String> rewriteLocalEnv(
            Map<String, String> env,
            Path execRoot,
            String localTmpRoot,
            String fallbackTmpDir,
            String productName) {
          return env;
        }
      };

  /**
   * Rewrites a {@code Spawn}'s the environment if necessary.
   *
   * @param env the Spawn's environment to rewrite
   * @param execRoot the path where the Spawn is executed
   * @param localTmpRoot an absolute path to a temp directory that the Spawn could use. Whether the
   *     particular implementation of {@link LocalEnvProvider} chooses to use this path, or {@code
   *     fallbackTmpDir}, or some other value, is up to the implementation. Typically the
   *     implementations will use {@code localTmpRoot}, or if empty then use the Bazel client's
   *     environment's TMPDIR/TMP/TEMP value (depending on the host OS), or if empty then use the
   *     {@code fallbackTmpDir} or some other value (typically "/tmp").
   * @param fallbackTmpDir an absolute path to a temp directory that the Spawn could use. Whether
   *     the particular implementation of {@link LocalEnvProvider} chooses to use this path, or
   *     {@code localTmpRoot}, or some other value, is up to the implementation. Typically the
   *     implementations will use {@code localTmpRoot}, or if empty then use the Bazel client's
   *     environment's TMPDIR/TMP/TEMP value (depending on the host OS), or if empty then use the
   *     {@code fallbackTmpDir} or some other value (typically "/tmp").
   * @param productName name of the Bazel binary, e.g. "bazel"
   */
  Map<String, String> rewriteLocalEnv(
      Map<String, String> env,
      Path execRoot,
      String localTmpRoot,
      String fallbackTmpDir,
      String productName)
      throws IOException;
}
