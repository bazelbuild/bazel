package com.google.devtools.build.lib.bazel.repository.decompressor;

import com.google.common.collect.ImmutableMap;
import com.google.common.io.ByteStreams;
import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.vfs.Path;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

/**
 * Common code for decompressing a single compressed file (compressor formats).
 *
 * <p>Apache Commons Compress calls all formats that compress a single stream of data compressor
 * formats while all formats that collect multiple entries inside a single (potentially compressed)
 * archive are archiver formats. This class handles the former, compressor formats.
 *
 * <p>It ignores the {@link DecompressorDescriptor#prefix()} setting because compressed files cannot
 * contain directories.
 */
public abstract class CompressedFunction implements Decompressor {

  protected abstract InputStream getDecompressorStream(BufferedInputStream compressedInputStream)
      throws IOException;

  /**
   * Returns the uncompressed file name (eg. file.gz -> file). Some compressors have
   * metadata that stores the original name. If that's the case, the original name is used (eg.
   * file.gz -> originalName). Only a basename + ext should be passed in for the compressedFileName.
   */
  protected abstract String getUncompressedFileName(
      InputStream in, final String compressedFileName);

  /**
   * Set custom file attributes, like last modified time, on the extracted file. Only certain
   * compressors support this.
   */
  protected void setFileAttributes(InputStream in, Path uncompressedFile) throws IOException {}

  // This is the same value as picked for .tar files, which appears to have worked well.
  private static final int BUFFER_SIZE = 32 * 1024;

  @Override
  public Path decompress(DecompressorDescriptor descriptor)
      throws InterruptedException, IOException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }

    ImmutableMap<String, String> renameFiles = descriptor.renameFiles();
    try (InputStream decompressorStream =
        getDecompressorStream(
            new BufferedInputStream(descriptor.archivePath().getInputStream(), BUFFER_SIZE))) {
      String entryName =
          getUncompressedFileName(decompressorStream, descriptor.archivePath().getBaseName());
      entryName = renameFiles.getOrDefault(entryName, entryName);
      Path filePath = descriptor.destinationPath().getRelative(entryName);
      filePath.getParentDirectory().createDirectoryAndParents();
      try (OutputStream out = filePath.getOutputStream()) {
        ByteStreams.copy(decompressorStream, out);
      }
      setFileAttributes(decompressorStream, filePath);
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }
    }
    return descriptor.destinationPath();
  }
}
