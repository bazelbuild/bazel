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

import com.google.devtools.build.lib.jni.JniLoader;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.common.options.OptionsParsingResult;

/** Various utilities related to UNIX processes. */
public final class ProcessUtilsServiceImpl implements ProcessUtilsService {

  @Override
  public void globalInit(OptionsParsingResult startupOptions) {
    ProcessUtilsService.registerJniService(this);
  }

  static {
    JniLoader.loadJni();
  }

  public ProcessUtilsServiceImpl() {}

  @Override
  public int getgid() {
    if (OS.getCurrent() == OS.WINDOWS) {
      throw new UnsupportedOperationException();
    }
    return getgidNative();
  }

  @Override
  public int getuid() {
    if (OS.getCurrent() == OS.WINDOWS) {
      throw new UnsupportedOperationException();
    }
    return getuidNative();
  }

  /**
   * Native wrapper around POSIX getgid(2).
   *
   * @return the real group ID of the current process.
   */
  private native int getgidNative();

  /**
   * Native wrapper around POSIX getuid(2).
   *
   * @return the real user ID of the current process.
   */
  private native int getuidNative();
}
