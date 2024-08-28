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
package com.google.devtools.build.android;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.base.Objects;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import javax.annotation.Nullable;

class UnwrittenMergedAndroidDataSubject extends Subject {

  private final UnwrittenMergedAndroidData actual;

  public UnwrittenMergedAndroidDataSubject(
      FailureMetadata failureMetadata, @Nullable UnwrittenMergedAndroidData subject) {
    super(failureMetadata, subject);
    this.actual = subject;
  }

  public void isEqualTo(UnwrittenMergedAndroidData expected) {
    UnwrittenMergedAndroidData subject = actual;
    if (!Objects.equal(subject, expected)) {
      if (subject == null) {
        assertThat(subject).isEqualTo(expected);
      }
      if (subject.getManifest() != null) {
        assertWithMessage("manifest")
            .that(subject.getManifest().toString())
            .isEqualTo(expected.getManifest().toString());
      }

      compareDataSets("resources", subject.getPrimary(), expected.getPrimary());
      compareDataSets("deps", subject.getTransitive(), expected.getTransitive());
    }
  }

  private void compareDataSets(
      String identifier, ParsedAndroidData subject, ParsedAndroidData expected) {
    assertWithMessage("Overwriting " + identifier)
        .that(subject.getOverwritingResources())
        .containsExactlyEntriesIn(expected.getOverwritingResources());
    assertWithMessage("Combining " + identifier)
        .that(subject.getCombiningResources())
        .containsExactlyEntriesIn(expected.getCombiningResources());
    assertWithMessage("Assets " + identifier)
        .that(subject.getAssets())
        .containsExactlyEntriesIn(expected.getAssets());
  }
}
