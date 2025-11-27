package com.google.devtools.build.lib.bazel.repository.decompressor;

import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import org.apache.commons.compress.compressors.bzip2.BZip2CompressorInputStream;
import org.apache.commons.compress.compressors.bzip2.BZip2Utils;

/**
 * Decompresses a bzip2 compressed file.
 */
public class Bz2Function extends CompressedFunction {
  public static final Decompressor INSTANCE = new Bz2Function();

  @Override
  protected InputStream getDecompressorStream(BufferedInputStream compressedInputStream)
      throws IOException {
    return new BZip2CompressorInputStream(compressedInputStream, true);
  }

  @Override
  protected String getUncompressedFileName(InputStream in, String compressedFileName) {
    return BZip2Utils.getUncompressedFileName(compressedFileName);
  }
}
