// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.ImmutableMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.CheckReturnValue;
import javax.annotation.Nullable;

/**
 * Opens the given list of input files and compute an index of all classes in them, to avoid
 * scanning all inputs over and over for each class to load. An indexed inputs can have a parent
 * that is firstly used when a file name is searched.
 */
public class IndexedInputs {

  private final ImmutableMap<String, InputFileProvider> inputFiles;

  /**
   * Parent {@link IndexedInputs} to use before to search a file name into this {@link
   * IndexedInputs}.
   */
  @Nullable private final IndexedInputs parent;

  /** Index a list of input files without a parent {@link IndexedInputs}. */
  public IndexedInputs(List<InputFileProvider> inputProviders) {
    this.parent = null;
    this.inputFiles = indexInputs(inputProviders);
  }

  /**
   * Create a new {@link IndexedInputs} with input files previously indexed and with a parent {@link
   * IndexedInputs}.
   */
  private IndexedInputs(
      ImmutableMap<String, InputFileProvider> inputFiles, IndexedInputs parentIndexedInputs) {
    this.parent = parentIndexedInputs;
    this.inputFiles = inputFiles;
  }

  /**
   * Create a new {@link IndexedInputs} with input files already indexed and with a parent {@link
   * IndexedInputs}.
   */
  @CheckReturnValue
  public IndexedInputs withParent(IndexedInputs parent) {
    checkState(this.parent == null);
    return new IndexedInputs(this.inputFiles, parent);
  }

  @Nullable
  public InputFileProvider getInputFileProvider(String filename) {
    checkArgument(filename.endsWith(".class"));

    if (parent != null) {
      InputFileProvider inputFileProvider = parent.getInputFileProvider(filename);
      if (inputFileProvider != null) {
        return inputFileProvider;
      }
    }

    return inputFiles.get(filename);
  }

  private ImmutableMap<String, InputFileProvider> indexInputs(
      List<InputFileProvider> inputProviders) {
    Map<String, InputFileProvider> indexedInputs = new LinkedHashMap<>();
    for (InputFileProvider inputProvider : inputProviders) {
      for (String relativePath : inputProvider) {
        if (relativePath.endsWith(".class") && !indexedInputs.containsKey(relativePath)) {
          indexedInputs.put(relativePath, inputProvider);
        }
      }
    }
    return ImmutableMap.copyOf(indexedInputs);
  }
}
