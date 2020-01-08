// Copyright 2019 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.android;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.hash.HashCode;
import com.google.common.jimfs.Jimfs;
import com.google.devtools.build.android.resources.Visibility;
import java.nio.file.FileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link DataValueFile}. */
@RunWith(JUnit4.class)
public final class DataValueFileTest {

  private final FileSystem fs = Jimfs.newFileSystem();

  @Test
  public void valueEquals_checkSourcePath() throws Exception {
    DataValueFile val1 =
        DataValueFile.of(
            Visibility.UNKNOWN,
            DataSource.of(DependencyInfo.UNKNOWN, fs.getPath("val1")),
            /*fingerprint=*/ null);
    DataValueFile val2a =
        DataValueFile.of(
            Visibility.UNKNOWN,
            DataSource.of(
                DependencyInfo.create("lib2a", DependencyInfo.DependencyType.UNKNOWN),
                fs.getPath("val2")),
            /*fingerprint=*/ null);
    DataValueFile val2b =
        DataValueFile.of(
            Visibility.UNKNOWN,
            DataSource.of(
                DependencyInfo.create("lib2b", DependencyInfo.DependencyType.UNKNOWN),
                fs.getPath("val2")),
            /*fingerprint=*/ null);

    assertThat(val1.valueEquals(val2a)).isFalse();
    assertThat(val2a.valueEquals(val2b)).isTrue();
  }

  @Test
  public void valueEquals_fallBackToFingerprint() throws Exception {
    DataValueFile val1 =
        DataValueFile.of(
            Visibility.UNKNOWN,
            DataSource.of(DependencyInfo.UNKNOWN, fs.getPath("asdf")),
            HashCode.fromInt(1));
    DataValueFile val2a =
        DataValueFile.of(
            Visibility.UNKNOWN,
            DataSource.of(DependencyInfo.UNKNOWN, fs.getPath("qwerty")),
            HashCode.fromInt(2));
    DataValueFile val2b =
        DataValueFile.of(
            Visibility.UNKNOWN,
            DataSource.of(DependencyInfo.UNKNOWN, fs.getPath("hunter2")),
            HashCode.fromInt(2));

    assertThat(val1.valueEquals(val2a)).isFalse();
    assertThat(val2a.valueEquals(val2b)).isTrue();
  }

  @Test
  public void valueEquals_checkVisibility() throws Exception {
    DataSource dataSource = DataSource.of(DependencyInfo.UNKNOWN, fs.getPath("x"));
    DataValueFile val1 = DataValueFile.of(Visibility.PRIVATE, dataSource, /*fingerprint=*/ null);
    DataValueFile val2a = DataValueFile.of(Visibility.PUBLIC, dataSource, /*fingerprint=*/ null);
    DataValueFile val2b = DataValueFile.of(Visibility.PUBLIC, dataSource, /*fingerprint=*/ null);

    assertThat(val1.valueEquals(val2a)).isFalse();
    assertThat(val2a.valueEquals(val2b)).isTrue();
  }
}
