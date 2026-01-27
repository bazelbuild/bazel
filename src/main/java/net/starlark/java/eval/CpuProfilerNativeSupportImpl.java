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
package net.starlark.java.eval;

import java.io.FileDescriptor;

/** Implementation of {@link CpuProfilerNativeSupport}. */
public final class CpuProfilerNativeSupportImpl implements CpuProfilerNativeSupport {

  static {
    JNI.load();
  }

  @Override
  public native boolean supported();

  @Override
  public native FileDescriptor createPipe();

  @Override
  public native boolean startTimer(long periodMicros);

  @Override
  public native void stopTimer();

  @Override
  public native int getThreadId();
}
