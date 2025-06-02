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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static java.lang.Math.min;

import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;

/**
 * Uses front compression to compress a list of paths.
 *
 * <p>This does not track the count of paths output and the client is responsible for doing so.
 */
final class PathCompression {

  static Compressor compressor() {
    return new Compressor();
  }

  static Decompressor decompressor() {
    return new Decompressor();
  }

  /**
   * Each call to the compressor will add an output line to the CodedOutputStream consisting of the
   * number of characters common with the beginning of the previous line followed by all the
   * characters that are different.
   */
  static class Compressor {
    private String previousPath = "";

    void compressTo(String path, CodedOutputStream out) throws IOException {
      int commonPrefixLength = getCommonPrefixLength(previousPath, path);
      String suffix = path.substring(commonPrefixLength);
      previousPath = path;
      out.writeInt32NoTag(commonPrefixLength);
      out.writeStringNoTag(suffix);
    }

    private static int getCommonPrefixLength(String s1, String s2) {
      int minLength = min(s1.length(), s2.length());
      int commonLength = 0;
      while (commonLength < minLength && s1.charAt(commonLength) == s2.charAt(commonLength)) {
        commonLength++;
      }
      return commonLength;
    }
  }

  /**
   * Each call to the decompressor will extract from a CodedInputStream a path compressed by
   * Compressor
   */
  static class Decompressor {
    private String previousPath = "";

    String decompressFrom(CodedInputStream codedIn) throws IOException {
      int commonPrefixLength = codedIn.readInt32();
      String suffix = codedIn.readString();
      String path = previousPath.substring(0, commonPrefixLength) + suffix;
      previousPath = path;
      return path;
    }
  }

  private PathCompression() {}
}
