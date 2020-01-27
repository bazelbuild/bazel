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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;
import org.mockito.Mockito;

/** Tests for {@link ArtifactsToOwnerLabels}. */
@RunWith(JUnit4.class)
public class ArtifactsToOwnerLabelsTest {
  @Test
  public void smoke() {
    Artifact label1Artifact = Mockito.mock(Artifact.class);
    Label label1 = Label.parseAbsoluteUnchecked("//label:one");
    Mockito.when(label1Artifact.getOwnerLabel()).thenReturn(label1);
    ArtifactsToOwnerLabels underTest =
        new ArtifactsToOwnerLabels.Builder().addArtifact(label1Artifact).build();
    assertThat(underTest.getOwners(label1Artifact)).containsExactly(label1);
    underTest = new ArtifactsToOwnerLabels.Builder().addArtifact(label1Artifact, label1).build();
    assertThat(underTest.getOwners(label1Artifact)).containsExactly(label1);
    underTest =
        new ArtifactsToOwnerLabels.Builder()
            .addArtifact(label1Artifact)
            .addArtifact(label1Artifact, label1)
            .build();
    assertThat(underTest.getOwners(label1Artifact)).containsExactly(label1);
    underTest =
        new ArtifactsToOwnerLabels.Builder()
            .addArtifact(label1Artifact, label1)
            .addArtifact(label1Artifact)
            .build();
    assertThat(underTest.getOwners(label1Artifact)).containsExactly(label1);
    Label label2 = Label.parseAbsoluteUnchecked("//label:two");
    underTest =
        new ArtifactsToOwnerLabels.Builder()
            .addArtifact(label1Artifact, label2)
            .addArtifact(label1Artifact)
            .build();
    assertThat(underTest.getOwners(label1Artifact)).containsExactly(label1, label2);
    underTest =
        new ArtifactsToOwnerLabels.Builder()
            .addArtifact(label1Artifact)
            .addArtifact(label1Artifact, label2)
            .build();
    assertThat(underTest.getOwners(label1Artifact)).containsExactly(label1, label2);
  }
}
