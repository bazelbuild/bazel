package com.google.devtools.build.lib.bazel.repository.decompressor;

import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import org.apache.commons.compress.compressors.gzip.GzipCompressorInputStream;
import org.apache.commons.compress.compressors.gzip.GzipParameters;
import org.apache.commons.compress.compressors.gzip.GzipUtils;

public class GzFunction extends CompressedFunction {
  public static final Decompressor INSTANCE = new GzFunction();

  @Override
  protected InputStream getDecompressorStream(BufferedInputStream compressedInputStream)
      throws IOException {
    return new GzipCompressorInputStream(compressedInputStream, true);
  }

  @Override
  protected String getUncompressedFileName(InputStream in, String compressedFileName) {
    String fileName = ((GzipCompressorInputStream) in).getMetaData().getFileName();
    if (fileName != null && !fileName.isBlank()) {
      // filename should be the simple basename + ext, but convert to a PathFragment and run
      // getBaseName to ensure that any path separators and uplevel references are dropped.
      return PathFragment.create(fileName).getBaseName().toString();
    }
    return GzipUtils.getUncompressedFileName(compressedFileName);
  }

  @Override
  protected void setFileAttributes(InputStream in, Path uncompressedFile) throws IOException {
    GzipParameters metaData = ((GzipCompressorInputStream) in).getMetaData();
    uncompressedFile.setLastModifiedTime(metaData.getModificationTime() * 1000);
  }
}
