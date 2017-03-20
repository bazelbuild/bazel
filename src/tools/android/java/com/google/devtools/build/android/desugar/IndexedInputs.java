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
package com.google.devtools.build.android.desugar;

import com.google.common.base.Preconditions;
import java.io.IOException;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Opens the given list of input files and compute an index of all classes in them, to avoid
 * scanning all inputs over and over for each class to load. An indexed inputs can have a parent
 * that is firstly used when a file name is searched.
 */
class IndexedInputs {

  private final Map<String, InputFileProvider> inputFiles = new HashMap<>();

  /** Parent indexed inputs to use before to search a file name into this indexed inputs. */
  @Nullable
  private final IndexedInputs parentIndexedInputs;

  /** Index a list of input files without a parent indexed inputs. */
  public IndexedInputs(List<InputFileProvider> inputProviders) throws IOException {
    this(inputProviders, null);
  }

  /**
   * Index a list of input files and set a parent indexed inputs that is firstly used during the
   * search of a file name.
   */
  public IndexedInputs(
      List<InputFileProvider> inputProviders, @Nullable IndexedInputs parentIndexedInputs)
      throws IOException {
    this.parentIndexedInputs = parentIndexedInputs;
    for (InputFileProvider inputProvider : inputProviders) {
      indexInput(inputProvider);
    }
  }

  @Nullable
  public InputFileProvider getInputFileProvider(String filename) {
    Preconditions.checkArgument(filename.endsWith(".class"));

    if (parentIndexedInputs != null) {
      InputFileProvider inputFileProvider = parentIndexedInputs.getInputFileProvider(filename);
      if (inputFileProvider != null) {
        return inputFileProvider;
      }
    }

    return inputFiles.get(filename);
  }

  private void indexInput(final InputFileProvider inputFileProvider) throws IOException {
    for (String relativePath : inputFileProvider) {
      if (relativePath.endsWith(".class") && !inputFiles.containsKey(relativePath)) {
        inputFiles.put(relativePath, inputFileProvider);
      }
    }
  }
}
