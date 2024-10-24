// Copyright 2022 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.starlarkbuildapi.core;

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static java.util.Arrays.stream;

import com.google.devtools.build.lib.cmdline.BazelCompileContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import net.starlark.java.eval.GuardedValue;
import net.starlark.java.eval.StarlarkSemantics;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class ContextGuardedValueTest {

  /**
   * We want to make sure the empty string doesn't result in "allow everything". That would be bad.
   * Most allowlists have an entry like ("", "tools/build_defs/lang") for usage within Google.
   */
  @Test
  public void emptyRepoAllowed_doesntMatchNonAllowed() throws Exception {
    assertNotAllowed("@mylang//bar:baz", "@//tools/lang");
  }

  @Test
  public void emptyRepoAllowed_matchesAllowed() throws Exception {
    assertAllowed("@//tools/lang", "@//tools/lang");
  }

  @Test
  public void workspaceRepo_matchesAllowedRepo() throws Exception {
    assertAllowed("@rules_foo//tools/lang", "@rules_foo//");
  }

  @Test
  public void workspaceRepo_doesntMatchCommonSubstr() throws Exception {
    assertNotAllowed("@my_rules_foo_helper//tools/lang", "@rules_foo//");
  }

  @Test
  public void bzlmodRepo_matchesStart() throws Exception {
    assertAllowed("@rules_foo+override//tools/lang", "@rules_foo//");
    assertAllowed("@rules_foo+1.2.3//tools/lang", "@rules_foo//");
  }

  @Test
  public void bzlmodRepo_matchesWithin() throws Exception {
    assertAllowed("@rules_lang+override+ext+foo_helper//tools/lang", "@foo_helper//");
  }

  @Test
  public void bzlmodRepo_doesntMatchCommonSubstr() throws Exception {
    assertNotAllowed("@rules_lang+override+ext+my_foo_helper_lib//tools/lang", "@foo_helper//");
  }

  @Test
  public void reposWithDotsDontMatch() throws Exception {
    assertNotAllowed("@my.lang//foo", "@my_lang//");
  }

  @Test
  public void verifySomeRealisticCases() throws Exception {
    // Python with workspace
    assertAllowed("@//tools/build_defs/python/private", "@//tools/build_defs/python");
    assertAllowed("@rules_python//python/private", "@rules_python//");

    // Python with bzlmod
    assertAllowed(
        "@rules_python+override+internal_deps+rules_python_internal//private", "@rules_python//");

    // CC with workspace
    assertAllowed("@//tools/build_defs/cc", "@//tools/build_defs/cc");
    assertNotAllowed("@rules_cc_helper//tools/build_defs/cc", "@rules_cc//");

    // CC with Bzlmod
    assertAllowed("@rules_cc+1.2.3+ext_name+local_cc_config//foo", "@local_cc_config//");
  }

  private Object createClientData(String callerLabelStr) {
    return BazelCompileContext.create(
        Label.parseCanonicalUnchecked(callerLabelStr), "unused_caller.bzl");
  }

  private GuardedValue createGuard(Object clientData, String... allowedLabelStrs) throws Exception {
    var allowed =
        stream(allowedLabelStrs)
            .map(
                labelStr -> {
                  try {
                    return PackageIdentifier.parse(labelStr);
                  } catch (LabelSyntaxException e) {
                    // We have to manually catch and re-throw this, otherwise Java is unhappy.
                    throw new RuntimeException(e);
                  }
                })
            .collect(toImmutableSet());

    return ContextGuardedValue.onlyInAllowedRepos(clientData, allowed);
  }

  private void assertAllowed(String callerLabelStr, String... allowedLabelStrs) throws Exception {
    var clientData = createClientData(callerLabelStr);
    var guard = createGuard(clientData, allowedLabelStrs);
    assertThat(guard.isObjectAccessibleUsingSemantics(StarlarkSemantics.DEFAULT, clientData))
        .isTrue();
  }

  private void assertNotAllowed(String callerLabelStr, String... allowedLabelStrs)
      throws Exception {
    var clientData = createClientData(callerLabelStr);
    var guard = createGuard(clientData, allowedLabelStrs);
    assertThat(guard.isObjectAccessibleUsingSemantics(StarlarkSemantics.DEFAULT, clientData))
        .isFalse();
  }
}
