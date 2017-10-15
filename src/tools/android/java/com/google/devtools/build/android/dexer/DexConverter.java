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

import com.android.dx.dex.file.DexFile;

/**
 * Converter from Java classes to corresponding standalone .dex files.
 */
class DexConverter {

  private final Dexing dexing;

  public DexConverter(Dexing dexing) {
    this.dexing = dexing;
  }

  public DexFile toDexFile(byte[] classfile, String classfilePath) {
    DexFile result = dexing.newDexFile();
    dexing.addToDexFile(result, Dexing.parseClassFile(classfile, classfilePath));
    return result;
  }

  public Dexing.DexingKey getDexingKey(byte[] classfile) {
    return dexing.getDexingKey(classfile);
  }
}
