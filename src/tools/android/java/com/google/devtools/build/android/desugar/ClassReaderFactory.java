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
package com.google.devtools.build.android.desugar;

import com.google.devtools.build.android.desugar.io.CoreLibraryRewriter;
import com.google.devtools.build.android.desugar.io.IndexedInputs;
import com.google.devtools.build.android.desugar.io.InputFileProvider;
import java.io.IOException;
import java.io.InputStream;
import javax.annotation.Nullable;
import org.objectweb.asm.ClassReader;

class ClassReaderFactory {
  private final IndexedInputs indexedInputs;
  private final CoreLibraryRewriter rewriter;

  public ClassReaderFactory(IndexedInputs indexedInputs, CoreLibraryRewriter rewriter) {
    this.rewriter = rewriter;
    this.indexedInputs = indexedInputs;
  }

  /**
   * Returns a reader for the given/internal/Class$Name if the class is defined in the wrapped input
   * and {@code null} otherwise. For simplicity this method turns checked into runtime exceptions
   * under the assumption that all classes have already been read once when this method is called.
   */
  @Nullable
  public ClassReader readIfKnown(String internalClassName) {
    String filename = rewriter.unprefix(internalClassName) + ".class";
    InputFileProvider inputFileProvider = indexedInputs.getInputFileProvider(filename);

    if (inputFileProvider != null) {
      try (InputStream bytecode = inputFileProvider.getInputStream(filename)) {
        // ClassReader doesn't take ownership and instead eagerly reads the stream's contents
        return rewriter.reader(bytecode);
      } catch (IOException e) {
        // We should've already read through all files in the Jar once at this point, so we don't
        // expect failures reading some files a second time.
        throw new IllegalStateException("Couldn't load " + internalClassName, e);
      }
    }

    return null;
  }

  /**
   * Returns {@code true} if the given given/internal/Class$Name is defined in the wrapped input.
   */
  public boolean isKnown(String internalClassName) {
    String filename = rewriter.unprefix(internalClassName) + ".class";
    return indexedInputs.getInputFileProvider(filename) != null;
  }
}
