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
package com.google.devtools.build.lib.packages.util;

import com.google.devtools.build.lib.packages.CachingPackageLocator;
import com.google.devtools.build.lib.packages.Globber;
import com.google.devtools.build.lib.packages.Preprocessor;
import com.google.devtools.build.lib.syntax.ParserInputSource;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import java.io.IOException;
import java.util.Set;

/**
 * Expands subinclude() statements, and returns an error if ERROR is
 * present in the end-result.  It does not run python, and is intended
 * for testing
 */
public class SubincludePreprocessor implements Preprocessor {
  /** Creates SubincludePreprocessor factories. */
  public static class FactorySupplier implements Preprocessor.Factory.Supplier {
    @Override
    public Factory getFactory(CachingPackageLocator loc, Path outputBase) {
      final SubincludePreprocessor preprocessor =
          new SubincludePreprocessor(outputBase.getFileSystem(), loc);
      return new Factory() {
        @Override
        public boolean isStillValid() {
          return true;
        }

        @Override
        public boolean considersGlobs() {
          return false;
        }

        @Override
        public Preprocessor getPreprocessor() {
          return preprocessor;
        }
      };
    }
  }

  /** Constructs a SubincludePreprocessor. Arguments are ignored. The class will be removed soon. */
  public SubincludePreprocessor(FileSystem fileSystem, CachingPackageLocator packageLocator) {}

  @Override
  public Preprocessor.Result preprocess(
      Path buildFilePath,
      byte[] buildFileBytes,
      String packageName,
      Globber globber,
      Set<String> ruleNames)
      throws IOException, InterruptedException {
    char content[] = FileSystemUtils.convertFromLatin1(buildFileBytes);
    return Preprocessor.Result.noPreprocessing(
        ParserInputSource.create(content, buildFilePath.asFragment()));
  }
}
