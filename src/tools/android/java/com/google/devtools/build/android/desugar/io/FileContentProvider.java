// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.desugar.io;

import static com.google.common.base.Preconditions.checkState;

import java.io.InputStream;
import javax.inject.Provider;

/**
 * A provider of an input stream with a file path label. The struct can be used to index byte code
 * files in a jar file, and serve as the reading source for the ASM library's class reader {@link
 * org.objectweb.asm.ClassReader}.
 */
public final class FileContentProvider<S extends InputStream> implements Provider<S> {

  private final String binaryPathName;
  private final Provider<S> inputStreamProvider;

  public FileContentProvider(String inArchiveBinaryPathName, Provider<S> inputStreamProvider) {
    checkState(
        !inArchiveBinaryPathName.startsWith("/"),
        "Expect inArchiveBinaryPathName is relative: (%s)",
        inArchiveBinaryPathName);
    this.binaryPathName = inArchiveBinaryPathName;
    this.inputStreamProvider = inputStreamProvider;
  }

  public String getBinaryPathName() {
    return binaryPathName;
  }

  @Override
  public S get() {
    return inputStreamProvider.get();
  }

  public boolean isClassFile() {
    return binaryPathName.endsWith(".class");
  }

  @Override
  public String toString() {
    return String.format("Binary Path: (%s)", binaryPathName);
  }
}
