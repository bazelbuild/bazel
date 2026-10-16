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

package com.google.devtools.build.lib.unix;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.unix.NativePosixFilesException.PosixError;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link NativePosixFilesException}. */
@RunWith(JUnit4.class)
public final class NativePosixFilesExceptionTest {

  @Test
  public void testErrorMapping() {
    assertThat(new NativePosixFilesException("msg", PosixError.ENOENT).getError())
        .isEqualTo(PosixError.ENOENT);
    assertThat(new NativePosixFilesException("msg", PosixError.EACCES).getError())
        .isEqualTo(PosixError.EACCES);
    assertThat(new NativePosixFilesException("msg", PosixError.ELOOP).getError())
        .isEqualTo(PosixError.ELOOP);
    assertThat(new NativePosixFilesException("msg", PosixError.ETIMEDOUT).getError())
        .isEqualTo(PosixError.ETIMEDOUT);
    assertThat(new NativePosixFilesException("msg", PosixError.OTHER).getError())
        .isEqualTo(PosixError.OTHER);
  }

  @Test
  public void testErrorMappingNullFallback() {
    assertThat(new NativePosixFilesException("msg", null).getError()).isEqualTo(PosixError.OTHER);
  }
}
