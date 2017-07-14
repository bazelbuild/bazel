// Copyright 2016 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.android.ideinfo;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.base.Joiner;
import com.google.devtools.build.lib.ideinfo.androidstudio.PackageManifestOuterClass.ArtifactLocation;
import com.google.devtools.common.options.OptionsParsingException;
import java.nio.file.Paths;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link ArtifactLocationConverter}.
 */
@RunWith(JUnit4.class)
public class ArtifactLocationConverterTest {

  private ArtifactLocationConverter converter;

  @Before
  public final void init() throws Exception  {
    converter = new ArtifactLocationConverter();
  }

  @Test
  public void testConverterSourceArtifact() throws Exception {
    ArtifactLocation parsed = converter.convert(
        Joiner.on(',').join("", "test.java")
    );
    assertThat(parsed)
        .isEqualTo(
            ArtifactLocation.newBuilder()
                .setRelativePath(Paths.get("test.java").toString())
                .setIsSource(true)
                .build());
  }

  @Test
  public void testConverterDerivedArtifact() throws Exception {
    ArtifactLocation parsed = converter.convert(
        Joiner.on(',').join("bin", "java/com/test.java")
    );
    assertThat(parsed)
        .isEqualTo(
            ArtifactLocation.newBuilder()
                .setRootExecutionPathFragment(Paths.get("bin").toString())
                .setRelativePath(Paths.get("java/com/test.java").toString())
                .setIsSource(false)
                .build());
  }

  @Test
  public void testConverterExternal() throws Exception {
    ArtifactLocation externalArtifact =
        converter.convert(Joiner.on(',').join("", "test.java", "1"));
    assertThat(externalArtifact)
        .isEqualTo(
            ArtifactLocation.newBuilder()
                .setRelativePath(Paths.get("test.java").toString())
                .setIsSource(true)
                .setIsExternal(true)
                .build());
    ArtifactLocation nonExternalArtifact =
        converter.convert(Joiner.on(',').join("", "test.java", "0"));
    assertThat(nonExternalArtifact)
        .isEqualTo(
            ArtifactLocation.newBuilder()
                .setRelativePath(Paths.get("test.java").toString())
                .setIsSource(true)
                .setIsExternal(false)
                .build());
  }

  @Test
  public void testInvalidFormatFails() throws Exception {
    assertFails("/root", ArtifactLocationConverter.INVALID_FORMAT);
    assertFails("/root,exec,rel,extra", ArtifactLocationConverter.INVALID_FORMAT);
  }

  private void assertFails(String input, String expectedError) {
    try {
      new ArtifactLocationConverter().convert(input);
      fail();
    } catch (OptionsParsingException e) {
      assertThat(e).hasMessage(expectedError);
    }
  }
}

