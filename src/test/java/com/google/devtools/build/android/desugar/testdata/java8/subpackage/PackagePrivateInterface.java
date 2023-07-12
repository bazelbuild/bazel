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
package com.google.devtools.build.android.desugar.testdata.java8.subpackage;

/** Package-private interface with default method. */
interface PackagePrivateInterface {

  /**
   * This field makes this interface need to be initialized. With the default methods, when this
   * interface is loaded, its initializer should also be run.
   *
   * <p>However, this test interface is different, as it is package-private. We need to to make sure
   * the desugared code does not trigger IllegalAccessError.
   *
   * <p>See b/38255926.
   */
  Integer VERSION = Integer.valueOf(0);

  default int m() {
    return 42;
  }
}
