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
package com.google.devtools.build.lib.skyframe;

import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.skyframe.SkyFunctionException.Transience;
import java.io.IOException;

/** Indicates some sort of IO error while dealing with a Starlark extension. */
public class ErrorReadingSkylarkExtensionException extends Exception {
  private final Transience transience;

  public ErrorReadingSkylarkExtensionException(BuildFileNotFoundException e) {
    super(e.getMessage(), e);
    this.transience = Transience.PERSISTENT;
  }

  public ErrorReadingSkylarkExtensionException(IOException e, Transience transience) {
    super(e.getMessage(), e);
    this.transience = transience;
  }

  Transience getTransience() {
    return transience;
  }
}
