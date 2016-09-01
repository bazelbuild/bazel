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
package com.google.devtools.build.lib.skylark;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.BuildView.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.SkylarkProviders;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.SkylarkClassObject;
import com.google.devtools.build.lib.packages.SkylarkClassObjectConstructor;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for Skylark.
 */
@RunWith(JUnit4.class)
public class SkylarkIntegrationTest extends AnalysisTestCase {
  protected boolean keepGoing() {
    return false;
  }

  @Test
  public void rulesReturningDeclaredProviders() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "my_provider = provider()",
        "def _impl(ctx):",
        "   return [my_provider(x = 1)]",
        "my_rule = rule(_impl)"
    );
    scratch.file(
        "test/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'r')"
    );

    AnalysisResult analysisResult = update("//test:r");
    ConfiguredTarget configuredTarget = analysisResult.getTargetsToBuild().iterator().next();
    SkylarkClassObjectConstructor.Key key = new SkylarkClassObjectConstructor.Key(
        Label.create(configuredTarget.getLabel().getPackageIdentifier(), "extension.bzl"),
        "my_provider");
    SkylarkProviders skylarkProviders = configuredTarget.getProvider(SkylarkProviders.class);
    assertThat(skylarkProviders).isNotNull();
    SkylarkClassObject declaredProvider = skylarkProviders.getDeclaredProvider(key);
    assertThat(declaredProvider).isNotNull();
    assertThat(declaredProvider.getConstructor().getKey()).isEqualTo(key);
    assertThat(declaredProvider.getValue("x")).isEqualTo(1);
  }

  @Test
  public void rulesReturningDeclaredProvidersCompatMode() throws Exception {
    scratch.file(
        "test/extension.bzl",
        "my_provider = provider()",
        "def _impl(ctx):",
        "   return struct(providers = [my_provider(x = 1)])",
        "my_rule = rule(_impl)"
    );
    scratch.file(
        "test/BUILD",
        "load(':extension.bzl', 'my_rule')",
        "my_rule(name = 'r')"
    );

    AnalysisResult analysisResult = update("//test:r");
    ConfiguredTarget configuredTarget = analysisResult.getTargetsToBuild().iterator().next();
    SkylarkClassObjectConstructor.Key key = new SkylarkClassObjectConstructor.Key(
        Label.create(configuredTarget.getLabel().getPackageIdentifier(), "extension.bzl"),
        "my_provider");
    SkylarkProviders skylarkProviders = configuredTarget.getProvider(SkylarkProviders.class);
    assertThat(skylarkProviders).isNotNull();
    SkylarkClassObject declaredProvider = skylarkProviders.getDeclaredProvider(key);
    assertThat(declaredProvider).isNotNull();
    assertThat(declaredProvider.getConstructor().getKey()).isEqualTo(key);
    assertThat(declaredProvider.getValue("x")).isEqualTo(1);
  }

  /**
   *  Same test with "keep going".
   */
  @RunWith(JUnit4.class)
  public static final class WithKeepGoing extends SkylarkIntegrationTest {
    @Override
    protected FlagBuilder defaultFlags() {
      return new FlagBuilder().with(Flag.KEEP_GOING);
    }

    @Override
    protected boolean keepGoing() {
      return true;
    }
  }

}
