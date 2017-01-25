// Copyright 2016 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android.dexer;

import com.android.dex.Dex;
import com.android.dx.dex.file.DexFile;
import java.io.IOException;

/**
 * Helper methods to write out {@code dx}'s {@link DexFile} objects.
 */
public class DexFiles {

  /**
   * Returns the {@link Dex} file resulting from writing out the given {@link DexFile}.
   */
  public static Dex toDex(DexFile dex) throws IOException {
    return new Dex(encode(dex));
  }

  /**
   * Serializes the given {@link DexFile} into {@code .dex}'s file format.
   */
  static byte[] encode(DexFile dex) throws IOException {
    return dex.toDex(null, false);
  }

  private DexFiles() {
  }
}
