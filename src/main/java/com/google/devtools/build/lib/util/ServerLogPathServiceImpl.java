// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.util;

import java.io.IOException;
import java.nio.file.Path;
import java.util.Optional;
import java.util.logging.Logger;

/** The {@link ServerLogPathService} implementation. */
public final class ServerLogPathServiceImpl implements ServerLogPathService {
  private static final Logger logger = Logger.getLogger(ServerLogPathServiceImpl.class.getName());

  @Override
  public Optional<String> getServerLogPath() throws IOException {
    // This must be called by the SC, as otherwise LogHandlerQuerier would resolve to a different
    // class when using a separate classloader for the LC.
    return LogHandlerQuerier.getConfiguredInstance()
        .getLoggerFilePath(logger)
        .map(Path::toAbsolutePath)
        .map(Object::toString);
  }
}
