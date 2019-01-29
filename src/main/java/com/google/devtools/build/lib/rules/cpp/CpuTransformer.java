// Copyright 2014 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.cpp;

import com.google.common.base.Function;
import com.google.common.base.Functions;

/** Catalog of possible cpu transformers. */
// TODO(b/120060214): Remove CpuTransformer
public enum CpuTransformer {
  IDENTITY {
    @Override
    public Function<String, String> getTransformer() {
      return Functions.identity();
    }
  },
  /**
   * NOT TO BE USED in Bazel.
   *
   * <p>Google-internal workaround.
   */
  FAKE {
    @Override
    public Function<String, String> getTransformer() {
      return FakeCPU::getRealCPU;
    }
  };

  public abstract Function<String, String> getTransformer();
}
