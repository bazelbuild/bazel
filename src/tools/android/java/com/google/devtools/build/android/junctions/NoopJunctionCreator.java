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
package com.google.devtools.build.android.junctions;

import java.io.IOException;
import java.nio.file.Path;
import javax.annotation.Nullable;

/** A no-op JunctionCreator implementation that just returns the input path. */
public final class NoopJunctionCreator implements JunctionCreator {
  @Nullable
  @Override
  public Path create(@Nullable Path path) throws IOException {
    return path;
  }

  @Override
  public void close() throws IOException {}
}
