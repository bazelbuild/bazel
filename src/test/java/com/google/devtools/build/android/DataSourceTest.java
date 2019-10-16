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

import com.google.common.jimfs.Jimfs;
import java.nio.file.FileSystem;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link DataSource}. */
@RunWith(JUnit4.class)
public final class DataSourceTest {

  private final FileSystem fs = Jimfs.newFileSystem();

  @Test
  public void combine_differentDependencyType_chooseCloser() throws Exception {
    DataSource source1 =
        DataSource.of(
            DependencyInfo.create("//pkg1", DependencyInfo.DependencyType.DIRECT),
            fs.getPath("res", "values", "foo"));
    DataSource source2 =
        DataSource.of(
            DependencyInfo.create("//pkg2", DependencyInfo.DependencyType.TRANSITIVE),
            fs.getPath("res", "values", "bar"));

    assertThat(source1.combine(source2)).isEqualTo(source1);
    assertThat(source2.combine(source1)).isEqualTo(source1);
  }

  @Test
  public void combine_equalDependencyType_chooseFirst() throws Exception {
    DataSource source1 =
        DataSource.of(
            DependencyInfo.create("//pkg1", DependencyInfo.DependencyType.TRANSITIVE),
            fs.getPath("res", "values", "foo"));
    DataSource source2 =
        DataSource.of(
            DependencyInfo.create("//pkg2", DependencyInfo.DependencyType.TRANSITIVE),
            fs.getPath("res", "values", "bar"));

    assertThat(source1.combine(source2)).isEqualTo(source1);
    assertThat(source2.combine(source1)).isEqualTo(source2);
  }

  @Test
  public void combine_equalDependencyType_chooseSourceWithValuesDir() throws Exception {
    DataSource source1 =
        DataSource.of(
            DependencyInfo.create("//pkg1", DependencyInfo.DependencyType.TRANSITIVE),
            fs.getPath("res", "values", "foo"));
    DataSource source2 =
        DataSource.of(
            DependencyInfo.create("//pkg2", DependencyInfo.DependencyType.TRANSITIVE),
            fs.getPath("res", "layout", "bar"));

    assertThat(source1.combine(source2)).isEqualTo(source1);
    assertThat(source2.combine(source1)).isEqualTo(source1);
  }
}
