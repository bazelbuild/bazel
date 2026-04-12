// Copyright 2026 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.bazel.repository.decompressor.DecompressorValue.Decompressor;
import java.io.BufferedInputStream;
import java.io.IOException;
import java.io.InputStream;
import org.apache.commons.compress.compressors.FileNameUtil;
import org.apache.commons.compress.compressors.brotli.BrotliCompressorInputStream;

/** Decompresses a brotli compressed file. */
public class BrFunction extends CompressedFunction {
  public static final Decompressor INSTANCE = new BrFunction();
  // Apache Commons Compress does not provide a readily available mapping of compressed ->
  // uncompressed filenames for brotli, so we make our own.
  static final FileNameUtil fileNameUtil = new FileNameUtil(ImmutableMap.of(".br", ""), ".br");

  @Override
  protected InputStream getDecompressorStream(BufferedInputStream compressedInputStream)
      throws IOException {
    return new BrotliCompressorInputStream(compressedInputStream);
  }

  @Override
  protected String getUncompressedFileName(InputStream in, String compressedFileName) {
    return fileNameUtil.getUncompressedFileName(compressedFileName);
  }
}
