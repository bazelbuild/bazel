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

package com.google.devtools.build.workspace.maven;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.events.StoredEventHandler;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link Rule}.
 */
@RunWith(JUnit4.class)
public class RuleTest {

  private StoredEventHandler handler;

  @Before
  public void createEventHandler() {
    handler = new StoredEventHandler();
  }

  @Test
  public void testUrl() throws Exception {
    Rule rule = new Rule("foo:bar:1.2.3");
    assertThat(rule.getUrl())
        .isEqualTo("https://repo1.maven.org/maven2/foo/bar/1.2.3/bar-1.2.3.pom");
    rule.setRepository("http://myrepo.com/foo/bar/1.2.3/bar-1.2.3.pom", handler);
    assertThat(handler.getEvents()).isEmpty();
    assertThat(rule.getUrl()).isEqualTo("http://myrepo.com/foo/bar/1.2.3/bar-1.2.3.pom");
  }
}
