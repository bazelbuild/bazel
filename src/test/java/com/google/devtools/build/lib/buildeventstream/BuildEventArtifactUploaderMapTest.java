// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.devtools.build.lib.buildeventstream;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.util.concurrent.Futures;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link BuildEventArtifactUploaderMap}. */
@RunWith(JUnit4.class)
public final class BuildEventArtifactUploaderMapTest {
  private BuildEventArtifactUploaderMap uploader;
  private BuildEventArtifactUploader noConversionUploader;

  @Before
  public void setUp() {
    noConversionUploader = files -> Futures.immediateFuture(PathConverter.NO_CONVERSION);
    uploader =
        new BuildEventArtifactUploaderMap.Builder()
            .add("a", BuildEventArtifactUploader.LOCAL_FILES_UPLOADER)
            .add("b", noConversionUploader)
            .build();
  }

  @Test
  public void testEmptyUploaders() throws Exception {
    BuildEventArtifactUploaderMap emptyUploader =
        new BuildEventArtifactUploaderMap.Builder().build();
    assertThat(emptyUploader.select(null))
        .isEqualTo(BuildEventArtifactUploader.LOCAL_FILES_UPLOADER);
  }

  @Test
  public void testAlphabeticalOrder() throws Exception {
    assertThat(uploader.select(null)).isEqualTo(BuildEventArtifactUploader.LOCAL_FILES_UPLOADER);
  }

  @Test
  public void testSelectByName() throws Exception {
    assertThat(uploader.select("b")).isEqualTo(noConversionUploader);
  }
}
