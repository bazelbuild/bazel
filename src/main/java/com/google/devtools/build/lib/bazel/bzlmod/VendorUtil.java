package com.google.devtools.build.lib.bazel.bzlmod;

import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.hash.HashCode;
import com.google.common.hash.Hasher;
import com.google.devtools.build.lib.bazel.repository.downloader.Checksum;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.Symlinks;

import java.io.IOException;
import java.io.UnsupportedEncodingException;
import java.net.URL;
import java.net.URLDecoder;
import java.util.Objects;
import java.util.Optional;


/**
 * Utility class for vendoring external repositories.
 */
public class VendorUtil {

  private final Path vendorDirectory;

  public VendorUtil(Path vendorDirectory) {
    this.vendorDirectory = vendorDirectory;
  }

  /**
   * Vendors the specified repositories under the vendor directory.
   *
   * @param externalRepoRoot The root directory of the external repositories.
   * @param reposToVendor The list of repositories to vendor.
   * @throws IOException if an I/O error occurs.
   */
  public void vendorRepos(
      Path externalRepoRoot,
      ImmutableList<RepositoryName> reposToVendor)
      throws IOException {
    if (!vendorDirectory.exists()) {
      vendorDirectory.createDirectoryAndParents();
    }

    for (RepositoryName repo : reposToVendor) {
      // Only re-vendor the repository if it is not up-to-date.
      if (!isRepoUpToDate(repo.getName(), externalRepoRoot)) {
        Path repoUnderVendor = vendorDirectory.getRelative(repo.getName());
        if (!repoUnderVendor.exists()) {
          repoUnderVendor.createDirectory();
        }
        FileSystemUtils.copyTreesBelow(
            externalRepoRoot.getRelative(repo.getName()), repoUnderVendor, Symlinks.NOFOLLOW);
        FileSystemUtils.copyFile(
            externalRepoRoot.getChild("@" + repo.getName() + ".marker"),
            vendorDirectory.getChild("@" + repo.getName() + ".marker"));
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
  public boolean isUrlVendored(URL url) throws UnsupportedEncodingException {
    return getVendorPathForURL(url).isFile();
  }

  /**
   * Vendors the registry URL with the specified content.
   *
   * @param url The registry URL to vendor.
   * @param content The content to write.
   * @throws IOException if an I/O error occurs.
   */
  public void vendorRegistryURL(URL url, byte[] content)
      throws IOException {
    Path outputPath = getVendorPathForURL(url);
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
  public byte[] readRegistryURL(URL url, Checksum checksum)
      throws IOException {
    byte[] content = FileSystemUtils.readContent(getVendorPathForURL(url));
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
   * Checks if the repository under vendor dir needs to be updated by comparing its marker file with the
   * one under <output_base>/external. This function assumes the marker file under <output_base>/external exists
   * and is up-to-date.
   *
   * @param repoName The name of the repository.
   * @param externalPath The root directory of the external repositories.
   * @return true if the repository is up-to-date, false otherwise.
   * @throws IOException if an I/O error occurs.
   */
  private boolean isRepoUpToDate(String repoName, Path externalPath)
      throws IOException {
    Path vendorMarkerFile = vendorDirectory.getChild("@" + repoName + ".marker");
    if (!vendorMarkerFile.exists()) {
      return false;
    }

    Path externalMarkerFile = externalPath.getChild("@" + repoName + ".marker");
    String vendorMarkerContent = FileSystemUtils.readContent(vendorMarkerFile, UTF_8);
    String externalMarkerContent = FileSystemUtils.readContent(externalMarkerFile, UTF_8);
    return Objects.equals(vendorMarkerContent, externalMarkerContent);
  }

  /**
   * Returns the vendor path for the given URL.
   *
   * The vendor path is constructed as follows:
   * <vendor_directory>/registry_cache/<host>/<path>
   *
   * The host name is case-insensitive, so it is converted to lowercase.
   * The path is case-sensitive, so it is left as is.
   * The port number is not included in the vendor path.
   *
   * Note that the vendor path may conflicts if two URLs only differ by the case or port number.
   * But this is unlikely to happen in practice, and conflicts are checked in VendorCommand.java.
   *
   * @param url The URL to get the vendor path for.
   * @return The vendor path.
   * @throws UnsupportedEncodingException if the URL decoding fails.
   */
  public Path getVendorPathForURL(URL url) throws UnsupportedEncodingException {
    String host = url.getHost().toLowerCase(); // Host names are case-insensitive
    String path = url.getPath();
    path = URLDecoder.decode(path, "UTF-8");
    if (path.startsWith("/")) {
      path = path.substring(1);
    }
    return vendorDirectory.getRelative("registry_cache").getRelative(host).getRelative(path);
  }
}
