// Copyright 2016 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.repository.ExternalPackageHelper;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import javax.annotation.Nullable;

/** SkyFunction for {@link LocalRepositoryLookupValue}s. */
public class LocalRepositoryLookupFunction implements SkyFunction {

  public LocalRepositoryLookupFunction(ExternalPackageHelper externalPackageHelper) {
    var unused = externalPackageHelper;
  }

  // Implementation note: Although LocalRepositoryLookupValue.NOT_FOUND exists, it should never be
  // returned from this method.
  @Nullable
  @Override
  public SkyValue compute(SkyKey skyKey, Environment env)
      throws SkyFunctionException, InterruptedException {
    // TODO: #22208, #21515 - Figure out what to do here.
    return LocalRepositoryLookupValue.mainRepository();
  }
}
