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
package com.google.devtools.build.lib.actions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.testing.EqualsTester;
import com.google.devtools.build.lib.util.Fingerprint;
import com.google.devtools.build.lib.vfs.FileStatus;
import java.io.IOException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link FileContentsProxy}. */
@RunWith(JUnit4.class)
public class FileContentsProxyTest {
  /** A simple implementation of FileStatus for testing. */
  private static final class InjectedStat implements FileStatus {
    private final long mtime;
    private final long ctime;
    private final long size;
    private final long nodeId;

    InjectedStat(long mtime, long ctime, long size, long nodeId) {
      this.mtime = mtime;
      this.ctime = ctime;
      this.size = size;
      this.nodeId = nodeId;
    }

    InjectedStat(long ctime, long nodeId) {
      this.ctime = ctime;
      this.mtime = ctime;
      this.nodeId = nodeId;
      this.size = 0;
    }

    @Override
    public boolean isFile() {
      return true;
    }

    @Override
    public boolean isSpecialFile() {
      return false;
    }

    @Override
    public boolean isDirectory() {
      return false;
    }

    @Override
    public boolean isSymbolicLink() {
      return false;
    }

    @Override
    public long getSize() {
      return size;
    }

    @Override
    public long getLastModifiedTime() {
      return mtime;
    }

    @Override
    public long getLastChangeTime() {
      return ctime;
    }

    @Override
    public long getNodeId() {
      return nodeId;
    }
  }

  @Test
  public void equalsAndHashCode() throws IOException {
    new EqualsTester()
        .addEqualityGroup(
            FileContentsProxy.create(new InjectedStat(1L, 2L)),
            FileContentsProxy.create(new InjectedStat(1L, 2L)))
        .addEqualityGroup(FileContentsProxy.create(new InjectedStat(1L, 4L)))
        .addEqualityGroup(FileContentsProxy.create(new InjectedStat(3L, 4L)))
        .addEqualityGroup(FileContentsProxy.create(new InjectedStat(-1L, -1L)))
        .testEquals();
  }

  @Test
  public void fingerprint() throws Exception {
    FileContentsProxy p1 =
        FileContentsProxy.create(
            new InjectedStat(/*mtime=*/1, /*ctime=*/2, /*size=*/3, /*nodeId=*/4));
    Fingerprint fingerprint = new Fingerprint();
    p1.addToFingerprint(fingerprint);
    assertThat(fingerprint.digestAndReset())
        .isEqualTo(new Fingerprint().addLong(2L).addLong(1L).addLong(4L).digestAndReset());
  }
}
