// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository.decompressor;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.bazel.repository.RepositoryFunctionException;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.io.IOException;
import java.nio.channels.ClosedByInterruptException;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.stream.Collectors;
import net.starlark.java.eval.Starlark;

/**
 * The contents of decompressed archive.
 */
public class DecompressorValue implements SkyValue {

  private static final ImmutableList<String> ZIP_FORMATS =
      ImmutableList.<String>builder()
          .add("zip")
          .add("jar")
          .add("war")
          .add("aar")
          .add("nupkg")
          .add("whl")
          .build();

  private static final ImmutableList<String> TAR_FORMATS =
      ImmutableList.<String>builder().add("tar").build();

  private static final ImmutableList<String> TAR_GZIP_FORMATS =
      ImmutableList.<String>builder().add("tar.gz").add("tgz").build();

  private static final ImmutableList<String> GZIP_FORMATS =
          ImmutableList.<String>builder().add("gz").build();

  private static final ImmutableList<String> TAR_XZ_FORMATS =
      ImmutableList.<String>builder().add("tar.xz").add("txz").build();

  private static final ImmutableList<String> XZ_FORMATS =
          ImmutableList.<String>builder().add("xz").build();

  private static final ImmutableList<String> TAR_ZST_FORMATS =
      ImmutableList.<String>builder().add("tar.zst").add("tzst").build();

  private static final ImmutableList<String> ZST_FORMATS =
          ImmutableList.<String>builder().add("zst").build();

  private static final ImmutableList<String> TAR_BZ2_FORMATS =
      ImmutableList.<String>builder().add("tar.bz2").add("tbz").build();

  private static final ImmutableList<String> BZ2_FORMATS =
          ImmutableList.<String>builder().add("bz2").build();

  private static final ImmutableList<String> AR_FORMATS =
      ImmutableList.<String>builder().add("ar").add("deb").build();

  private static final ImmutableList<String> SEVENZ_FORMATS =
      ImmutableList.<String>builder().add("7z").build();

  // List of supported compressor format file extensions with their corresponding Decompressor
  // instance. The order here is intentional and is the order in which a decompressor is searched
  // for.
  private static final ImmutableList<Pair<ImmutableList<String>, Decompressor>> supportedFormats =
      ImmutableList.<Pair<ImmutableList<String>, Decompressor>>builder()
          .add(Pair.of(ZIP_FORMATS, ZipDecompressor.INSTANCE))
          .add(Pair.of(TAR_FORMATS, TarFunction.INSTANCE))
          .add(Pair.of(TAR_GZIP_FORMATS, TarGzFunction.INSTANCE))
          .add(Pair.of(GZIP_FORMATS, GzFunction.INSTANCE))
          .add(Pair.of(TAR_XZ_FORMATS, TarXzFunction.INSTANCE))
          .add(Pair.of(XZ_FORMATS, XzFunction.INSTANCE))
          .add(Pair.of(TAR_ZST_FORMATS, TarZstFunction.INSTANCE))
          .add(Pair.of(ZST_FORMATS, ZstFunction.INSTANCE))
          .add(Pair.of(TAR_BZ2_FORMATS, TarBz2Function.INSTANCE))
          .add(Pair.of(BZ2_FORMATS, Bz2Function.INSTANCE))
          .add(Pair.of(AR_FORMATS, ArFunction.INSTANCE))
          .add(Pair.of(SEVENZ_FORMATS, SevenZDecompressor.INSTANCE))
          .build();

  /**
   * Returns a human-readable string of supported decompressor extensions separated by commas.
   *
   * <p>The resulting string looks like:
   *
   * <p><code>
   * [prefix][extension][suffix], [prefix][extension2][suffix] [conjunction] [prefix][extension3][suffix]
   * </p></code>
   *
   * <p>Examples:
   *
   * <ul>
   *   <li>No prefix/suffix and with the conjunction "and": <code>jar, zip, whl, tgz and ar</code>
   *   <li>Dot prefix and conjunction "or": <code>.jar, .zip, .whl, .tgz or .ar</code>
   *   <li>Quote+dot prefix, quote suffix and conjunction "or": <code>
   *       `.jar`, `.zip`, `.whl`, `.tgz` or `.ar`</code>
   * </ul>
   */
  public static String readableSupportedFormats(String prefix, String suffix, String conjunction) {
    List<String> allExtensions = allSupportedExtensions(prefix, suffix);

    String commaSeparatedExtensions =
        allExtensions.subList(0, allExtensions.size() - 1).stream()
            .collect(Collectors.joining(", "));

    return commaSeparatedExtensions + " " + conjunction + " " + allExtensions.getLast();
  }

  public static List<String> allSupportedExtensions(String prefix, String suffix) {
    return supportedFormats.stream()
        .map(format -> format.first)
        .flatMap(List::stream)
        .map(type -> prefix + type + suffix)
        .collect(Collectors.toList());
  }

  /** Implementation of a decompression algorithm. */
  public interface Decompressor {

    /** Exception reporting about absence of an expected prefix in an archive. */
    class CouldNotFindPrefixException extends IOException {

      CouldNotFindPrefixException(String prefix, Set<String> availablePrefixes) {

        super(CouldNotFindPrefixException.prepareErrorMessage(prefix, availablePrefixes));
      }

      private static String prepareErrorMessage(String prefix, Set<String> availablePrefixes) {
        String error = "Prefix \"" + prefix + "\" was given, but not found in the archive. ";
        String suggestion = "Here are possible prefixes for this archive: ";
        String suggestionBody = "";

        if (availablePrefixes.isEmpty()) {
          suggestion =
              "We could not find any directory in this archive"
                  + " (maybe there is no need for `strip_prefix`?)";
        } else {
          // Add a list of possible suggestion wrapped with `"` and separated by `, `.
          suggestionBody = "\"" + String.join("\", \"", availablePrefixes) + "\".";
        }

        return error + suggestion + suggestionBody;
      }

      public static Optional<String> maybeMakePrefixSuggestion(PathFragment pathFragment) {
        if (!pathFragment.isMultiSegment()) {
          return Optional.empty();
        }
        return Optional.of(pathFragment.getSegment(0));
      }
    }

    Path decompress(DecompressorDescriptor descriptor)
        throws IOException, RepositoryFunctionException, InterruptedException;
  }

  private final Path directory;

  public DecompressorValue(Path repositoryPath) {
    directory = repositoryPath;
  }

  @Override
  public boolean equals(Object other) {
    return this == other
        || (other instanceof DecompressorValue decompressorValue
            && directory.equals(decompressorValue.directory));
  }

  @Override
  public int hashCode() {
    return directory.hashCode();
  }

  @VisibleForTesting
  static Decompressor getDecompressor(Path archivePath) throws RepositoryFunctionException {
    String baseName = archivePath.getBaseName();
    // Return the corresponding decompressor if the archive's basename ends with a matching
    // extension. Eg. If the file ends in .tar.gz or .tgz, use the TarGzFunction decompressor.
    for (Pair<ImmutableList<String>, Decompressor> format : supportedFormats) {
      ImmutableList<String> fileExtensions = format.first;
      Decompressor decompressor = format.second;
      if (fileExtensions.stream().map(type -> "." + type).anyMatch(ext -> baseName.endsWith(ext))) {
        return decompressor;
      }
    }

    throw new RepositoryFunctionException(
        Starlark.errorf(
            "Expected a file with a %s suffix (got %s)",
            readableSupportedFormats(/* prefix= */ ".", /* suffix= */ "", /* conjunction= */ "or"),
            archivePath),
        Transience.PERSISTENT);
  }

  @CanIgnoreReturnValue
  public static Path decompress(DecompressorDescriptor descriptor)
      throws RepositoryFunctionException, InterruptedException {
    try {
      return getDecompressor(descriptor.archivePath()).decompress(descriptor);
    } catch (ClosedByInterruptException e) {
      throw new InterruptedException();
    } catch (IOException e) {
      Path destinationDirectory = descriptor.archivePath().getParentDirectory();
      throw new RepositoryFunctionException(
          new IOException(
              String.format(
                  "Error extracting %s to %s: %s",
                  descriptor.archivePath(), destinationDirectory, e.getMessage())),
          Transience.TRANSIENT);
    }
  }
}
