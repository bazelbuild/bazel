// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import com.beust.jcommander.Parameter;
import com.beust.jcommander.Parameters;

/**
 * Test class that has a boolean JCommander parameter.
 *
 * <p>To be used with AndroidOptionsUtilsTest.
 */
@Parameters(separators = "= ")
class TestOptions {
  @Parameter(names = "--foo", description = "foo", arity = 1)
  public boolean foo = true;

  @Parameter(names = "--normalize", description = "a flag with 'no' in the name", arity = 1)
  public boolean normalize = true;
}
