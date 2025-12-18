// Copyright 2025 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.unix;

import com.google.devtools.build.lib.runtime.BlazeService;
import java.util.concurrent.atomic.AtomicReference;

/** Service for various UNIX process utilities. */
public interface ProcessUtilsService extends BlazeService {
  static final AtomicReference<ProcessUtilsService> service = new AtomicReference<>();

  public static void registerJniService(ProcessUtilsService service) {
    ProcessUtilsService.service.set(service);
  }

  public static ProcessUtilsService getService() {
    return service.get();
  }

  /**
   * Returns the real group ID of the current process.
   *
   * @throws UnsatisfiedLinkError when JNI is not available.
   * @throws UnsupportedOperationException on operating systems where this call is not implemented.
   */
  int getgid();

  /**
   * Returns the real user ID of the current process.
   *
   * @throws UnsatisfiedLinkError when JNI is not available.
   * @throws UnsupportedOperationException on operating systems where this call is not implemented.
   */
  int getuid();
}
