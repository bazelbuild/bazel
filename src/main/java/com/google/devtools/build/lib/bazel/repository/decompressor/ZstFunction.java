package com.google.devtools.build.lib.bazel.repository.decompressor;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import org.apache.commons.compress.compressors.FileNameUtil;
import org.apache.commons.compress.compressors.zstandard.ZstdCompressorInputStream;

public class ZstFunction extends CompressedFunction {
  public static final Decompressor INSTANCE = new ZstFunction();
  // Apache Commons Compress does not provide a readily available mapping of compressed ->
  // uncompressed filenames for Zst, so we make our own.
  static FileNameUtil fileNameUtil = new FileNameUtil(ImmutableMap.of(".zst", ""), ".zst");

  @Override
  protected InputStream getDecompressorStream(BufferedInputStream compressedInputStream)
      throws IOException {
    return new ZstdCompressorInputStream(compressedInputStream);
  }

  @Override
  protected String getUncompressedFileName(InputStream in, String compressedFileName) {
    return fileNameUtil.getUncompressedFileName(compressedFileName);
  }
}
