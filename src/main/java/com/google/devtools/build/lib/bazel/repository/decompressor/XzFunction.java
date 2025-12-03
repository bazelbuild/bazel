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

import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import org.apache.commons.compress.compressors.xz.XZCompressorInputStream;
import org.apache.commons.compress.compressors.xz.XZUtils;

/** Decompresses an xz (LZMA) compressed file. */
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
   */
  @Override
  protected InputStream getDecompressorStream(BufferedInputStream compressedInputStream)
      throws IOException {
    return new XZCompressorInputStream(compressedInputStream);
  }

  @Override
  protected String getUncompressedFileName(InputStream in, String compressedFileName) {
    return XZUtils.getUncompressedFileName(compressedFileName);
  }
}
