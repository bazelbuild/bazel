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

import com.android.resources.ResourceType;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.jimfs.Jimfs;
import com.google.devtools.build.android.resources.Visibility;
import java.nio.file.FileSystem;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PlaceholderRTxtWriter}. */
@RunWith(JUnit4.class)
public final class PlaceholderRTxtWriterTest {

  private final FileSystem fs = Jimfs.newFileSystem();

  @Test
  public void acceptSimpleResource() throws Exception {
    Path rTxt = fs.getPath("r.txt");
    PlaceholderRTxtWriter rTxtWriter = PlaceholderRTxtWriter.create(rTxt);

    rTxtWriter.acceptSimpleResource(
        DependencyInfo.UNKNOWN, Visibility.UNKNOWN, ResourceType.ARRAY, "x");
    rTxtWriter.acceptSimpleResource(
        DependencyInfo.UNKNOWN, Visibility.UNKNOWN, ResourceType.STRING, "y");
    rTxtWriter.acceptSimpleResource(
        DependencyInfo.UNKNOWN, Visibility.UNKNOWN, ResourceType.STRING, "z");
    rTxtWriter.flush();

    assertThat(Files.readAllLines(rTxt))
        .containsExactly("int array x 0", "int string y 0", "int string z 0");
  }

  @Test
  public void acceptStyleableResource() throws Exception {
    Path rTxt = fs.getPath("r.txt");
    PlaceholderRTxtWriter rTxtWriter = PlaceholderRTxtWriter.create(rTxt);

    rTxtWriter.acceptStyleableResource(
        DependencyInfo.UNKNOWN,
        Visibility.UNKNOWN,
        FullyQualifiedName.of(
            FullyQualifiedName.DEFAULT_PACKAGE,
            /*qualifiers=*/ ImmutableList.of(),
            ResourceType.STYLEABLE,
            "x"),
        ImmutableMap.of(
            FullyQualifiedName.of(
                FullyQualifiedName.DEFAULT_PACKAGE,
                ImmutableList.of(),
                ResourceType.STRING,
                "attr1"),
            true,
            FullyQualifiedName.of("android", ImmutableList.of(), ResourceType.LAYOUT, "attr2"),
            true));
    rTxtWriter.flush();

    assertThat(Files.readAllLines(rTxt))
        .containsExactly(
            "int[] styleable x { 0 }",
            "int styleable x_android_attr2 0",
            "int styleable x_attr1 0");
  }
}
