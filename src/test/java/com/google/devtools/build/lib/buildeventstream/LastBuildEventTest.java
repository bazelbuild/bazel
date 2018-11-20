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

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import java.util.Collection;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link LastBuildEvent}. */
@RunWith(JUnit4.class)
public class LastBuildEventTest {

  @Test
  public void testForwardsReferencedLocalFilesCall() {
    LocalFile localFile = new LocalFile(null, LocalFileType.FAILED_TEST_OUTPUT);
    LastBuildEvent event = new LastBuildEvent(new BuildEvent() {
      @Override
      public BuildEventId getEventId() {
        return null;
      }

      @Override
      public Collection<BuildEventId> getChildrenEvents() {
        return null;
      }

      @Override
      public Collection<LocalFile> referencedLocalFiles() {
        return ImmutableList.of(localFile);
      }

      @Override
      public BuildEventStreamProtos.BuildEvent asStreamProto(BuildEventContext context) {
        return null;
      }
    });
    assertThat(event.referencedLocalFiles()).containsExactly(localFile);
  }
}
