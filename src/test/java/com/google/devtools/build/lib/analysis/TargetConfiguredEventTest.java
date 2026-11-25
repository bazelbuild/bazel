// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEvent;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.TestSize;
import com.google.devtools.build.lib.rules.AliasConfiguredTarget;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests covering {@link TargetConfiguredEvent} properties. */
@RunWith(JUnit4.class)
public class TargetConfiguredEventTest extends BuildViewTestCase {

  private void assertTestSize(String label, TestSize testSize) throws Exception {
    TargetConfiguredEvent targetConfiguredEvent =
        new TargetConfiguredEvent(getTarget(label), getTargetConfiguration(), /* actual= */ null);
    // The converters are never used in the method so it is okay to set to null here.
    BuildEvent buildEvent = targetConfiguredEvent.asStreamProto(/* converters= */ null);
    assertThat(buildEvent.getConfigured().getTestSize()).isEqualTo(testSize);
  }

  @Test
  public void testTargetConfiguredEvent_isSmallTest() throws Exception {
    scratch.file(
        "p/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='small_test', size='small', srcs=['small_test.sh'])");
    assertTestSize("//p:small_test", TestSize.SMALL);
  }

  @Test
  public void testTargetConfiguredEvent_isMediumTest() throws Exception {
    scratch.file(
        "p/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='medium_test', size='medium', srcs=['medium_test.sh'])");
    assertTestSize("//p:medium_test", TestSize.MEDIUM);
  }

  @Test
  public void testTargetConfiguredEvent_isLargeTest() throws Exception {
    scratch.file(
        "p/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='large_test', size='large', srcs=['large_test.sh'])");
    assertTestSize("//p:large_test", TestSize.LARGE);
  }

  @Test
  public void testTargetConfiguredEvent_isEnormousTest() throws Exception {
    scratch.file(
        "p/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='enormous_test', size='enormous', srcs=['enormous_test.sh'])");
    assertTestSize("//p:enormous_test", TestSize.ENORMOUS);
  }

  @Test
  public void testTargetConfiguredEvent_isNotTest() throws Exception {
    scratch.file(
        "p/BUILD",
        "load('@rules_java//java:java_library.bzl', 'java_library')",
        "java_library(name='foo', srcs=['Foo.java'])");
    assertTestSize("//p:foo", TestSize.UNKNOWN);
  }

  @Test
  public void testTargetConfiguredEvent_buildEventId() throws Exception {
    scratch.file(
        "p/BUILD",
        "load('//test_defs:foo_test.bzl', 'foo_test')",
        "foo_test(name='test', size='small', srcs=['test.sh'])");
    TargetConfiguredEvent targetConfiguredEvent =
        new TargetConfiguredEvent(
            getTarget("//p:test"), /* configuration= */ null, /* actual= */ null);
    BuildEventId.TargetConfiguredId expectedBuildEventId =
        BuildEventId.TargetConfiguredId.newBuilder().setLabel("//p:test").build();
    assertThat(targetConfiguredEvent.getEventId().getTargetConfigured())
        .isEqualTo(expectedBuildEventId);
  }

  @Test
  public void testTargetConfiguredEvent_aliasTarget() throws Exception {
    String buildFile =
        """
        load('//test_defs:foo_test.bzl', 'foo_test')
        foo_test(name='test', size='small', srcs=['test.sh'])
        alias(name='alias2', actual=':test')
        alias(name='alias1', actual=':alias2')
        """;
    scratch.file("p/BUILD", buildFile);
    ConfiguredTarget alias1 = getConfiguredTarget("//p:alias1");
    assertThat(alias1).isInstanceOf(AliasConfiguredTarget.class);
    TargetConfiguredEvent targetConfiguredEvent =
        new TargetConfiguredEvent(
            getTarget("//p:alias1"), /* configuration= */ null, alias1.getActual().getLabel());
    BuildEvent buildEvent = targetConfiguredEvent.asStreamProto(/* converters= */ null);
    assertThat(buildEvent.getConfigured().getActual()).isEqualTo("//p:test");
  }
}
