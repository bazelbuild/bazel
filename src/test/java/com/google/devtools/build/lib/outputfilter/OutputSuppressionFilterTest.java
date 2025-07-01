// Copyright 2024 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.outputfilter;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventContext;
import com.google.devtools.build.lib.events.EventKind;
import net.starlark.java.syntax.Location;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link OutputSuppressionFilter}. */
@RunWith(JUnit4.class)
public class OutputSuppressionFilterTest {

  @Test
  public void testSmoke() {
    OutputSuppressionFilter filter =
        new OutputSuppressionFilter(
            ImmutableList.of(
                "count:1 package:@@rules_jvm_external The maven repository 'maven' is used in two"
                    + " different bazel modules"));

    Event event =
        Event.of(
                EventKind.DEBUG,
                "The maven repository 'maven' is used in two different bazel modules")
            .withProperty(
                EventContext.class,
                EventContext.builder().setPackage("@@rules_jvm_external").build());

    assertThat(filter.showOutput(event)).isFalse();
  }

  @Test
  public void testStarlarkEvent() {
    OutputSuppressionFilter filter =
        new OutputSuppressionFilter(
            ImmutableList.of(
                "package:@@rules_jvm_external The maven repository 'maven' is used in two"
                    + " different bazel modules"));

    Location location =
        new Location(
            "/private/var/tmp/_bazel_keir/123/external/rules_jvm_external/some/path/to/file.bzl",
            0,
            0);
    Event event =
        Event.of(
            EventKind.DEBUG,
            location,
            "The maven repository 'maven' is used in two different bazel modules");

    assertThat(filter.showOutput(event)).isFalse();
  }

  @Test
  public void testPathMatching() {
    OutputSuppressionFilter filter =
        new OutputSuppressionFilter(
            ImmutableList.of("path:.*/external/protobuf/.* The py_proto_library macro is deprecated"));

    Location location =
        new Location(
            "/private/var/tmp/_bazel_keir/123/external/protobuf/some/path/to/file.bzl", 0, 0);
    Event event =
        Event.of(EventKind.DEBUG, location, "The py_proto_library macro is deprecated");

    assertThat(filter.showOutput(event)).isFalse();
  }

  @Test
  public void testRealWorldExamples() {
    OutputSuppressionFilter filter =
        new OutputSuppressionFilter(
            ImmutableList.of(
                "path:.*/src/conditions/BUILD select.. on cpu is deprecated",
                "path:.*/external/\\+async_profiler_repos\\+async_profiler_macos/BUILD.bazel depends on deprecated target",
                "path:.*/external/c-ares\\+/BUILD.bazel depends on deprecated target",
                "path:.*/src/test/java/com/google/devtools/build/lib/testutil/BUILD depends on deprecated target",
                "path:.*/external/protobuf\\+/protobuf.bzl The py_proto_library macro is deprecated"));

    // Test case 1: select() deprecation
    Location location1 = new Location("/Users/keir/wrk/bazel/src/conditions/BUILD", 202, 15);
    Event event1 =
        Event.of(
            EventKind.WARNING,
            location1,
            "in config_setting rule //src/conditions:windows_arm64_flag: select() on cpu is"
                + " deprecated. Use platform constraints instead:"
                + " https://bazel.build/docs/configurable-attributes#platforms.");
    assertThat(filter.showOutput(event1)).isFalse();

    // Test case 2: depends on deprecated target (async_profiler)
    Location location2 =
        new Location(
            "/private/var/tmp/_bazel_keir/2a787604a869aba515a82a8f5f5252a4/external/+async_profiler_repos+async_profiler_macos/BUILD.bazel",
            4,
            10);
    Event event2 =
        Event.of(
            EventKind.WARNING,
            location2,
            "in _copy_file rule"
                + " @@+async_profiler_repos+async_profiler_macos//:libasyncProfiler: target '"
                + " @@+async_profiler_repos+async_profiler_macos//:libasyncProfiler' depends on"
                + " deprecated target '"
                + " @@bazel_tools//src/conditions:host_windows_x64_constraint': No longer used by"
                + " Bazel and will be removed in the future. Migrate to"
                + " toolchains or define your own version of this setting.");
    assertThat(filter.showOutput(event2)).isFalse();

    // Test case 3: depends on deprecated target (c-ares)
    Location location3 =
        new Location(
            "/private/var/tmp/_bazel_keir/2a787604a869aba515a82a8f5f5252a4/external/c-ares+/BUILD.bazel",
            48,
            10);
    Event event3 =
        Event.of(
            EventKind.WARNING,
            location3,
            "in _copy_file rule @@c-ares+//:ares_config_h: target ' @@c-ares+//:ares_config_h'"
                + " depends on deprecated target '"
                + " @@bazel_tools//src/conditions:host_windows_x64_constraint': No longer used by"
                + " Bazel and will be removed in the future. Migrate to toolchains or define your"
                + " own version of this setting.");
    assertThat(filter.showOutput(event3)).isFalse();

    // Test case 4: py_proto_library deprecation
    Location location4 =
        new Location(
            "/private/var/tmp/_bazel_keir/2a787604a869aba515a82a8f5f5252a4/external/protobuf+/protobuf.bzl",
            650,
            10);
    Event event4 =
        Event.of(
            EventKind.DEBUG,
            location4,
            "The py_proto_library macro is deprecated and will be removed in the 30.x release."
                + " switch to the rule defined by rules_python or the one in"
                + " bazel/py_proto_library.bzl.");
    assertThat(filter.showOutput(event4)).isFalse();

    // Test case 5: depends on deprecated target in testutil
    Location location5 =
        new Location("/Users/keir/wrk/bazel/src/test/java/com/google/devtools/build/lib/testutil/BUILD", 281, 11);
    Event event5 =
        Event.of(
            EventKind.WARNING,
            location5,
            "in _write_file rule //src/test/java/com/google/devtools/build/lib/testutil:gen_rules_cc_repo_name: "
                + "target '//src/test/java/com/google/devtools/build/lib/testutil:gen_rules_cc_repo_name' "
                + "depends on deprecated target '@@bazel_tools//src/conditions:host_windows_x64_constraint': "
                + "No longer used by Bazel and will be removed in the future. "
                + "Migrate to toolchains or define your own version of this setting.");
    assertThat(filter.showOutput(event5)).isFalse();
  }
}
