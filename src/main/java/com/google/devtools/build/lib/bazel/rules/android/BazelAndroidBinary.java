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
package com.google.devtools.build.lib.bazel.rules.android;

import com.google.devtools.build.lib.bazel.rules.cpp.BazelCppSemantics;
import com.google.devtools.build.lib.bazel.rules.java.BazelJavaSemantics;
import com.google.devtools.build.lib.rules.android.AndroidBinary;
import com.google.devtools.build.lib.rules.android.AndroidSemantics;
import com.google.devtools.build.lib.rules.cpp.CppSemantics;
import com.google.devtools.build.lib.rules.java.JavaSemantics;

/**
 * Bazel-specific configured target factory for {@code android_binary}.
 */
public class BazelAndroidBinary extends AndroidBinary {
  @Override
  protected JavaSemantics createJavaSemantics() {
    return BazelJavaSemantics.INSTANCE;
  }

  @Override
  protected AndroidSemantics createAndroidSemantics() {
    return BazelAndroidSemantics.INSTANCE;
  }

  @Override
  protected CppSemantics createCppSemantics() {
    return BazelCppSemantics.CPP;
  }
}
