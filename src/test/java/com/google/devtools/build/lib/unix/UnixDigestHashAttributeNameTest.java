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
package com.google.devtools.build.lib.unix;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.vfs.DigestHashFunction;
import com.google.devtools.build.lib.vfs.DigestUtils;
import com.google.devtools.build.lib.vfs.FileSystem;
import com.google.devtools.build.lib.vfs.FileSystemTest;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.lib.vfs.SyscallCache;
import org.junit.Test;

/** Test for {@link com.google.devtools.build.lib.unix.UnixFileSystem#getFastDigest}. */
public class UnixDigestHashAttributeNameTest extends FileSystemTest {
  private static final byte[] FAKE_DIGEST = {
    0x18, 0x5f, 0x3d, 0x33, 0x22, 0x71, 0x7e, 0x25,
    0x55, 0x61, 0x26, 0x0c, 0x03, 0x6b, 0x2e, 0x26,
    0x43, 0x06, 0x7c, 0x30, 0x4e, 0x3a, 0x51, 0x20,
    0x07, 0x71, 0x76, 0x48, 0x26, 0x38, 0x19, 0x69,
  };

  @Override
  protected FileSystem getFreshFileSystem(DigestHashFunction digestHashFunction) {
    return new FakeAttributeFileSystem(digestHashFunction);
  }

  @Test
  public void testFoo() throws Exception {
    // Instead of actually trying to access this file, a call to getxattr() should be made. We
    // intercept this call and return a fake extended attribute value, thereby causing the checksum
    // computation to be skipped entirely.
    assertThat(DigestUtils.getDigestWithManualFallback(absolutize("myfile"), SyscallCache.NO_CACHE))
        .isEqualTo(FAKE_DIGEST);
  }

  private class FakeAttributeFileSystem extends UnixFileSystem {
    public FakeAttributeFileSystem(DigestHashFunction hashFunction) {
      super(hashFunction, "user.checksum.sha256");
    }

    @Override
    public byte[] getxattr(PathFragment path, String name, boolean followSymlinks) {
      assertThat(path).isEqualTo(absolutize("myfile").asFragment());
      assertThat(name).isEqualTo("user.checksum.sha256");
      assertThat(followSymlinks).isTrue();
      return FAKE_DIGEST;
    }
  }
}
