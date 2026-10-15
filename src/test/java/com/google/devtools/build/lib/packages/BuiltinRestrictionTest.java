// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.packages;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.BuiltinRestriction.externalRepoAllowlistEntry;
import static com.google.devtools.build.lib.packages.BuiltinRestriction.mainRepoAllowlistEntry;

import com.google.devtools.build.lib.cmdline.Label;
import com.google.testing.junit.testparameterinjector.TestParameterInjector;
import com.google.testing.junit.testparameterinjector.TestParameters;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;

/** Tests for {@link BuiltinRestriction}. */
@RunWith(TestParameterInjector.class)
public final class BuiltinRestrictionTest {
  private static final BuiltinRestriction.Allowlist ALLOWLIST =
      BuiltinRestriction.Allowlist.of(
          mainRepoAllowlistEntry("allowed"),
          externalRepoAllowlistEntry("rules_cc", "pkg"),
          externalRepoAllowlistEntry("bazel_tools", "tools"));

  @Test
  @TestParameters("{label: '//allowed:defs.bzl', moduleRepoName: '', allowed: true}")
  @TestParameters("{label: '//allowed/sub:defs.bzl', moduleRepoName: '', allowed: true}")
  @TestParameters("{label: '//allowed_other:defs.bzl', moduleRepoName: '', allowed: false}")
  @TestParameters("{label: '//pkg:defs.bzl', moduleRepoName: 'rules_cc', allowed: true}")
  @TestParameters("{label: '//pkg:defs.bzl', moduleRepoName: 'custom', allowed: false}")
  @TestParameters("{label: '//pkg:defs.bzl', moduleRepoName: '', allowed: false}")
  @TestParameters("{label: '//pkg_other:defs.bzl', moduleRepoName: 'rules_cc', allowed: false}")
  @TestParameters("{label: '@@rules_cc+//pkg:defs.bzl', moduleRepoName: 'custom', allowed: true}")
  @TestParameters("{label: '@@rules_cc+1.0//pkg/sub:defs.bzl', moduleRepoName: '', allowed: true}")
  @TestParameters(
      "{label: '@@rules_cc++ext+repo//pkg:defs.bzl', moduleRepoName: '', allowed: true}")
  @TestParameters("{label: '@@rules_cc_extra+//pkg:defs.bzl', moduleRepoName: '', allowed: false}")
  @TestParameters("{label: '@@other+//pkg:defs.bzl', moduleRepoName: 'rules_cc', allowed: false}")
  @TestParameters("{label: '@@other+//allowed:defs.bzl', moduleRepoName: '', allowed: false}")
  @TestParameters(
      "{label: '@@+ext+repo//allowed:defs.bzl', moduleRepoName: 'rules_cc', allowed: false}")
  @TestParameters("{label: '@@bazel_tools//tools:defs.bzl', moduleRepoName: null, allowed: true}")
  @TestParameters("{label: '@@bazel_tools//other:defs.bzl', moduleRepoName: null, allowed: false}")
  @TestParameters("{label: '@@_builtins//other:defs.bzl', moduleRepoName: null, allowed: true}")
  public void allowlistMatches(String label, @Nullable String moduleRepoName, boolean allowed)
      throws Exception {
    assertThat(
            BuiltinRestriction.isNotAllowed(Label.parseCanonical(label), moduleRepoName, ALLOWLIST))
        .isEqualTo(!allowed);
  }
}
