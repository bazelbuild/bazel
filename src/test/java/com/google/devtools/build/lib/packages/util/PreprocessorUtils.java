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
import com.google.devtools.build.lib.packages.Preprocessor;

import javax.annotation.Nullable;

/**
 * Testing utilities for {@link Preprocessor}.
 */
public class PreprocessorUtils {

  public static class MutableFactorySupplier implements Preprocessor.Factory.Supplier {

    @Nullable private Preprocessor preprocessor;
    private boolean valid = true;
    private Factory factory = new Factory();

    public MutableFactorySupplier(@Nullable Preprocessor preprocessor) {
      this.preprocessor = preprocessor;
    }

    public void inject(@Nullable Preprocessor preprocessor) {
      this.valid = false;
      this.preprocessor = preprocessor;
    }

    @Override
    public Factory getFactory(CachingPackageLocator loc) {
      valid = true;
      return factory;
    }

    private class Factory implements Preprocessor.Factory {

      @Override
      public boolean isStillValid() {
        return valid;
      }

      @Override
      public boolean considersGlobs() {
        return false;
      }

      @Override
      @Nullable
      public Preprocessor getPreprocessor() {
        return preprocessor;
      }
    }
  }
}
