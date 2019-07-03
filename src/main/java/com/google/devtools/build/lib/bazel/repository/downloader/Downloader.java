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

package com.google.devtools.build.lib.bazel.repository.downloader;

import com.google.common.base.MoreObjects;
import com.google.common.base.Optional;
import com.google.common.base.Strings;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.net.URI;
import java.net.URL;
import java.util.List;
import java.util.Map;

/**
 * Bazel file downloader.
 */
public abstract class Downloader {
  /**
   * Downloads file to disk and returns path.
   *
   * <p>If the checksum and path to the repository cache is specified, attempt to load the file from
   * the {@link RepositoryCache}. If it doesn't exist, proceed to download the file and load it into
   * the cache prior to returning the value.
   *
   * @param urls list of mirror URLs with identical content
   * @param checksum valid checksum which is checked, or empty to disable
   * @param type extension, e.g. "tar.gz" to force on downloaded filename, or empty to not do this
   * @param output destination filename if {@code type} is <i>absent</i>, otherwise output directory
   * @param eventHandler CLI progress reporter
   * @param clientEnv environment variables in shell issuing this command
   * @param repo the name of the external repository for which the file was fetched; used only for
   *     reporting
   * @throws IllegalArgumentException on parameter badness, which should be checked beforehand
   * @throws IOException if download was attempted and ended up failing
   * @throws InterruptedException if this thread is being cast into oblivion
   */
  public abstract Path download(
    List<URL> urls,
    Map<URI, Map<String, String>> authHeaders,
    Optional<Checksum> checksum,
    String canonicalId,
    Optional<String> type,
    Path output,
    ExtendedEventHandler eventHandler,
    Map<String, String> clientEnv,
    String repo)
    throws IOException, InterruptedException;

  protected static Path getDownloadDestination(URL url, Optional<String> type, Path output) {
    if (!type.isPresent()) {
      return output;
    }
    String basename =
        MoreObjects.firstNonNull(
            Strings.emptyToNull(PathFragment.create(url.getPath()).getBaseName()),
            "temp");
    if (!type.get().isEmpty()) {
      String suffix = "." + type.get();
      if (!basename.endsWith(suffix)) {
        basename += suffix;
      }
    }
    return output.getRelative(basename);
  }
}
