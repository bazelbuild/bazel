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
package com.google.devtools.build.lib.remote;

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.remote.common.CacheNotFoundException;
import java.io.IOException;

/**
 * Exception which represents a collection of IOExceptions for the purpose
 * of distinguishing remote communication exceptions from those which occur
 * on filesystems locally. This exception serves as a trace point for the actual
 * download, so that the intented operation can be observed in a stack, with all
 * constituent exceptions available for observation.
 */
class DownloadException extends IOException {
  // true since no empty DownloadException is ever thrown
  private boolean allCacheNotFoundException = true;

  DownloadException() {
  }

  DownloadException(IOException e) {
    add(e);
  }

  void add(IOException e) {
    if (allCacheNotFoundException) {
      allCacheNotFoundException = e instanceof CacheNotFoundException;
    }
    super.addSuppressed(e);
  }

  boolean onlyCausedByCacheNotFoundException() {
    return allCacheNotFoundException;
  }
}

