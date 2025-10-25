package com.google.devtools.build.lib.bazel.repository.decompressor;

import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import org.apache.commons.compress.compressors.xz.XZCompressorInputStream;
import org.apache.commons.compress.compressors.xz.XZUtils;

public class XzFunction extends CompressedFunction {
  public static final Decompressor INSTANCE = new XzFunction();

  /**
   * Uses {@link XZCompressorInputStream} from Apache Commons Compress to decompress.
   *
   * <p>Why not use {@link org.tukaani.xz.XZInputStream} which is used in {@link TarXzFunction}? The
   * Apache Commons Compress libraries are wrappers around org.tukaani.xz.XZInputStream, so they
   * should be the same. Since we also use {@link
   * org.apache.commons.compress.compressors.xz.XZUtils}, we keep consistency and use the Apache
   * wrapper consistently in this class.
   *
   * @see <a
   *     href="https://commons.apache.org/proper/commons-compress/apidocs/org/apache/commons/compress/compressors/xz/package-summary.html">javadoc</a>
   * @param compressedInputStream
   * @return
   * @throws IOException
   */
  @Override
  protected InputStream getDecompressorStream(BufferedInputStream compressedInputStream)
      throws IOException {
    return new XZCompressorInputStream(compressedInputStream, true);
  }

  @Override
  protected String getUncompressedFileName(InputStream in, String compressedFileName) {
    return XZUtils.getUncompressedFileName(compressedFileName);
  }
}
