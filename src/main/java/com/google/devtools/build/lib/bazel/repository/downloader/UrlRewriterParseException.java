// Copyright 2021 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.repository.downloader;

import javax.annotation.Nullable;
import net.starlark.java.syntax.Location;

/** An {@link Exception} thrown when failed to parse {@link UrlRewriterConfig}. */
public class UrlRewriterParseException extends Exception {

  @Nullable private final Location location;

  public UrlRewriterParseException(String message) {
    this(message, /* location= */ null);
  }

  public UrlRewriterParseException(String message, @Nullable Location location) {
    super(message);
    this.location = location;
  }

  @Nullable
  public Location getLocation() {
    return location;
  }
}
