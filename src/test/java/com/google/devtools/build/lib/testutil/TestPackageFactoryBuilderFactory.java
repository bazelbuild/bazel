// Copyright 2020 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.testutil;

import com.google.devtools.build.lib.packages.BuilderFactoryForTesting;

/** Holds the {@link BuilderFactoryForTesting} to be used by tests. */
public class TestPackageFactoryBuilderFactory {

  private TestPackageFactoryBuilderFactory() {}

  /** Returns the instance to be used by tests. */
  public static final BuilderFactoryForTesting getInstance() {
    return PackageFactoryBuilderFactoryForBazelUnitTests.INSTANCE;
  }

}
