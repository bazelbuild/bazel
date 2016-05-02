// Copyright 2015 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.bazel.repository;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.bazel.repository.DecompressorValue.Decompressor;

/**
 * Creates a repository for a random file.
 */
public class FileDecompressor extends JarDecompressor {
  public static final Decompressor INSTANCE = new FileDecompressor();

  private FileDecompressor() {
  }

  @Override
  protected String getPackageName() {
    return "file";
  }

  @Override
  protected String createBuildFile(String baseName) {
    return Joiner.on("\n")
        .join(
            "filegroup(",
            "    name = 'file',",
            "    srcs = ['" + baseName + "'],",
            "    visibility = ['//visibility:public']",
            ")");
  }
}
