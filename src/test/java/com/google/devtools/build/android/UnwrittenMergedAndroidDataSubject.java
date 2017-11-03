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

import com.google.common.base.Objects;
import com.google.common.truth.FailureMetadata;
import com.google.common.truth.Subject;
import javax.annotation.Nullable;

class UnwrittenMergedAndroidDataSubject
    extends Subject<UnwrittenMergedAndroidDataSubject, UnwrittenMergedAndroidData> {

  static final Subject.Factory<UnwrittenMergedAndroidDataSubject, UnwrittenMergedAndroidData>
      FACTORY =
          new Subject.Factory<UnwrittenMergedAndroidDataSubject, UnwrittenMergedAndroidData>() {
            @Override
            public UnwrittenMergedAndroidDataSubject createSubject(
                FailureMetadata fs, UnwrittenMergedAndroidData that) {
              return new UnwrittenMergedAndroidDataSubject(fs, that);
            }
          };

  public UnwrittenMergedAndroidDataSubject(
      FailureMetadata failureStrategy, @Nullable UnwrittenMergedAndroidData subject) {
    super(failureStrategy, subject);
  }

  public void isEqualTo(UnwrittenMergedAndroidData expected) {
    UnwrittenMergedAndroidData subject = getSubject();
    if (!Objects.equal(subject, expected)) {
      if (subject == null) {
        assertThat(subject).isEqualTo(expected);
      }
      if (subject.getManifest() != null) {
        assertThat(subject.getManifest().toString())
            .named("manifest")
            .isEqualTo(expected.getManifest().toString());
      }

      compareDataSets("resources", subject.getPrimary(), expected.getPrimary());
      compareDataSets("deps", subject.getTransitive(), expected.getTransitive());
    }
  }

  private void compareDataSets(
      String identifier, ParsedAndroidData subject, ParsedAndroidData expected) {
    assertThat(subject.getOverwritingResources())
        .named("Overwriting " + identifier)
        .containsExactlyEntriesIn(expected.getOverwritingResources());
    assertThat(subject.getCombiningResources())
        .named("Combining " + identifier)
        .containsExactlyEntriesIn(expected.getCombiningResources());
    assertThat(subject.getAssets())
        .named("Assets " + identifier)
        .containsExactlyEntriesIn(expected.getAssets());
  }
}
