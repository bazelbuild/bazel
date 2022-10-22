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
package com.google.devtools.build.android.dexer.testdata.innerclasses;

interface Inner {
    public void bar();
}

public class OuterClass {
  public void foo() {

    // Syntethic lambdas should be packed together.
    Inner in1 = () -> {
    };
    Inner in2 = () -> {
    };
    Inner in3 = () -> {
    };
    Inner in4 = () -> {
    };
    Inner in5 = () -> {
    };
    Inner in6 = () -> {
    };
  }

  // Plain inner classes may be packed in a different shard.
  private static class InnerClass0 {
  }
  private static class InnerClass1 {
  }
  private static class InnerClass2 {
  }
  private static class InnerClass3 {
  }
  private static class InnerClass4 {
  }
  private static class InnerClass5 {
  }
}
