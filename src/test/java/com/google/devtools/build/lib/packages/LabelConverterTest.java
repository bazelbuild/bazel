// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.vfs.PathFragment;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Test of {@link LabelConverter}. */
@RunWith(JUnit4.class)
public class LabelConverterTest {

  @Test
  public void convertLabel() throws Exception {
    PackageIdentifier basePackage = PackageIdentifier.create("quux", PathFragment.create("baz"));
    LabelConverter converter =
        new LabelConverter(
            basePackage,
            RepositoryMapping.create(
                ImmutableMap.of("orig_repo", RepositoryName.create("new_repo")),
                RepositoryName.MAIN));
    assertThat(converter.convert("@orig_repo//foo:bar"))
        .isEqualTo(Label.parseCanonical("@new_repo//foo:bar"));
    assertThat(converter.convert("//foo:bar")).isEqualTo(Label.parseCanonical("@quux//foo:bar"));
    assertThat(converter.convert(":bar")).isEqualTo(Label.parseCanonical("@quux//baz:bar"));
  }
}
