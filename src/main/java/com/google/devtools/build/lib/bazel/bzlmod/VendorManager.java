// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.bzlmod;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.devtools.build.lib.bazel.repository.RepositoryUtils;
import com.google.devtools.build.lib.bazel.repository.cache.LocalRepoContentsCache;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URI;
import java.net.URLDecoder;
import java.util.Locale;
import java.util.Objects;

/** Class to manage the vendor directory. */
public class VendorManager {

  private static final String REGISTRIES_DIR = "_registries";

  public static final PathFragment EXTERNAL_ROOT_SYMLINK_NAME =
      PathFragment.create("bazel-external");

  private final Path vendorDirectory;

  public VendorManager(Path vendorDirectory) {
    this.vendorDirectory = vendorDirectory;
  }

  /**
   * Vendors the specified repositories under the vendor directory.
   *
   * <p>TODO(pcloudy): Parallelize vendoring repos
   *
   * @param externalRepoRoot The root directory of the external repositories.
   * @param workspace The workspace directory.
   * @param reposToVendor The list of repositories to vendor.
   * @throws IOException if an I/O error occurs.
   */
  public void vendorRepos(
      Path externalRepoRoot, Path workspace, ImmutableList<RepositoryName> reposToVendor)
      throws IOException {
    if (!vendorDirectory.exists()) {
      vendorDirectory.createDirectoryAndParents();
    }

    for (RepositoryName repo : reposToVendor) {
      try (SilentCloseable c =
          Profiler.instance().profile(ProfilerTask.REPOSITORY_VENDOR, repo.toString())) {
        Path repoUnderExternal = externalRepoRoot.getChild(repo.getName());
        Path repoUnderVendor = vendorDirectory.getChild(repo.getName());
        // This could happen when running the vendor command twice without changing anything.
        if (repoUnderExternal.isSymbolicLink()
            && repoUnderExternal.resolveSymbolicLinks().equals(repoUnderVendor)) {
          continue;
        }

        Path markerUnderExternal = externalRepoRoot.getChild(repo.getMarkerFileName());
        Path markerUnderVendor = vendorDirectory.getChild(repo.getMarkerFileName());
        // If the marker file doesn't exist under outputBase/external, then the repo is either local
        // (which cannot be in this case since local repos aren't vendored) or in the repo contents
        // cache.
        boolean isCached = !markerUnderExternal.exists();
        Path actualMarkerFile;
        if (isCached) {
          Path cacheRepoDir = repoUnderExternal.resolveSymbolicLinks();
          actualMarkerFile =
              cacheRepoDir.replaceName(
                  cacheRepoDir.getBaseName() + LocalRepoContentsCache.RECORDED_INPUTS_SUFFIX);
        } else {
          actualMarkerFile = markerUnderExternal;
        }

        // At this point, the repo should exist under external dir, but check if the vendor src is
        // already up-to-date.
        if (isRepoUpToDate(markerUnderVendor, actualMarkerFile)) {
          continue;
        }

        // Actually vendor the repo. If the repo is cached, copy it; otherwise move it.
        // 1. Clean up existing marker file and vendor dir.
        markerUnderVendor.delete();
        repoUnderVendor.deleteTree();
        repoUnderVendor.createDirectory();
        // 2. Copy/move the marker file to a temporary one under vendor dir.
        Path temporaryMarker = vendorDirectory.getChild(repo.getMarkerFileName() + ".tmp");
        if (isCached) {
          FileSystemUtils.copyFile(actualMarkerFile, temporaryMarker);
        } else {
          FileSystemUtils.moveFile(actualMarkerFile, temporaryMarker);
        }
        // 3. Move/copy the external repo to vendor dir. Note that, in the "move" case, it's fine if
        // this step fails or is interrupted, because the marker file under external is gone anyway.
        if (isCached) {
          FileSystemUtils.copyTreesBelow(repoUnderExternal.resolveSymbolicLinks(), repoUnderVendor);
        } else {
          try {
            repoUnderExternal.renameTo(repoUnderVendor);
          } catch (IOException e) {
            FileSystemUtils.moveTreesBelow(repoUnderExternal, repoUnderVendor);
          }
        }
        // 4. Re-plant symlinks pointing to a Bazel-managed path to a relative path to make sure the
        // vendor src keep working after being moved (including to a different checkout of the
        // workspace) or used with a different output base. We assume that a given vendor directory
        // is only used with one output base at a time.
        Path externalSymlink = vendorDirectory.getRelative(EXTERNAL_ROOT_SYMLINK_NAME);
        FileSystemUtils.ensureSymbolicLink(externalSymlink, externalRepoRoot);
        RepositoryUtils.replantSymlinks(
            repoUnderVendor, workspace, externalRepoRoot, EXTERNAL_ROOT_SYMLINK_NAME);
        // 5. Rename the temporary marker file after the move/copy is done.
        temporaryMarker.renameTo(markerUnderVendor);
        // 6. Leave a symlink in external dir to keep things working.
        repoUnderExternal.deleteTree();
        FileSystemUtils.ensureSymbolicLink(repoUnderExternal, repoUnderVendor);
      }
    }
  }

  /**
   * Checks if the given URL is vendored.
   *
   * @param url The URL to check.
   * @return true if the URL is vendored, false otherwise.
   * @throws UnsupportedEncodingException if the URL decoding fails.
   */
  public boolean isUrlVendored(URI url) throws UnsupportedEncodingException {
    return getVendorPathForUrl(url).isFile();
  }

  /**
   * Vendors the registry URL with the specified content.
   *
   * @param url The registry URL to vendor.
   * @param content The content to write.
   * @throws IOException if an I/O error occurs.
   */
  public void vendorRegistryUrl(URI url, byte[] content) throws IOException {
    Path outputPath = getVendorPathForUrl(url);
    Objects.requireNonNull(outputPath.getParentDirectory()).createDirectoryAndParents();
    FileSystemUtils.writeContent(outputPath, content);
  }

  /**
   * Reads the content of the registry URL and verifies its checksum.
   *
   * @param url The registry URL to read.
   * @param checksum The checksum to verify.
   * @return The content of the registry URL.
   * @throws IOException if an I/O error occurs or the checksum verification fails.
   */
  public byte[] readRegistryUrl(URI url, Checksum checksum) throws IOException {
    byte[] content = FileSystemUtils.readContent(getVendorPathForUrl(url));
    Hasher hasher = checksum.getKeyType().newHasher();
    hasher.putBytes(content);
    HashCode actual = hasher.hash();
    if (!checksum.getHashCode().equals(actual)) {
      throw new IOException(
          String.format(
              "Checksum was %s but wanted %s",
              checksum.emitOtherHashInSameFormat(actual),
              checksum.emitOtherHashInSameFormat(checksum.getHashCode())));
    }
    return content;
  }

  /**
   * Checks if the repository under vendor dir is up-to-date by comparing its marker file with the
   * one under <output_base>/external. This function assumes the marker file under
   * <output_base>/external exists and is up-to-date.
   *
   * @param markerUnderVendor The marker file path under vendor dir
   * @param markerUnderExternal The marker file path under external dir
   * @return true if the repository is up-to-date, false otherwise.
   * @throws IOException if an I/O error occurs.
   */
  private boolean isRepoUpToDate(Path markerUnderVendor, Path markerUnderExternal)
      throws IOException {
    if (!markerUnderVendor.exists()) {
      return false;
    }
    String vendorMarkerContent = FileSystemUtils.readContent(markerUnderVendor, UTF_8);
    String externalMarkerContent = FileSystemUtils.readContent(markerUnderExternal, UTF_8);
    return Objects.equals(vendorMarkerContent, externalMarkerContent);
  }

  /**
   * Returns the vendor path for the given URL.
   *
   * <p>The vendor path is constructed as follows: <vendor_directory>/registry_cache/<host>/<path>
   *
   * <p>The host name is case-insensitive, so it is converted to lowercase. The path is
   * case-sensitive, so it is left as is. The port number is not included in the vendor path.
   *
   * <p>Note that the vendor path may conflict if two URLs only differ by the case or port number.
   * But this is unlikely to happen in practice, and conflicts are checked in VendorCommand.java.
   *
   * @param url The URL to get the vendor path for.
   * @return The vendor path.
   * @throws UnsupportedEncodingException if the URL decoding fails.
   */
  public Path getVendorPathForUrl(URI url) throws UnsupportedEncodingException {
    String host = url.getHost().toLowerCase(Locale.ROOT); // Host names are case-insensitive
    String path = url.getPath();
    path = URLDecoder.decode(path, "UTF-8");
    if (path.startsWith("/")) {
      path = path.substring(1);
    }
    return vendorDirectory.getRelative(REGISTRIES_DIR).getRelative(host).getRelative(path);
  }
}
