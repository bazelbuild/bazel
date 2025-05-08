// Copyright 2023 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config.transitions;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.EventKind;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** {@link ComparingTransition} tests. */
@RunWith(JUnit4.class)
public final class ComparingTransitionTest extends BuildViewTestCase {
  @Test
  public void sameOutputs() throws Exception {
    PatchTransition trans1 = (options, eventHandler) -> options.underlying();
    PatchTransition trans2 = (options, eventHandler) -> options.underlying();
    BuildOptionsView fromOptions =
        new BuildOptionsView(
            targetConfig.getOptions(), targetConfig.getOptions().getFragmentClasses());

    var unused =
        new ComparingTransition(trans1, "trans1", trans2, "trans2", b -> true)
            .patch(fromOptions, reporter);
    String msg = Iterables.getOnlyElement(eventCollector.filtered(EventKind.INFO)).getMessage();

    assertThat(msg).contains("unique fragments in trans1 mode: none");
    assertThat(msg).contains("unique fragments in trans2 mode: none");
    assertThat(msg).contains("total option differences: 0");
  }

  @Test
  public void differentNativeFlag() throws Exception {
    PatchTransition trans1 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).stampBinaries = true;
          return toOptions;
        };
    PatchTransition trans2 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).stampBinaries = false;
          return toOptions;
        };
    BuildOptionsView fromOptions =
        new BuildOptionsView(
            targetConfig.getOptions(), targetConfig.getOptions().getFragmentClasses());

    var unused =
        new ComparingTransition(trans1, "trans1", trans2, "trans2", b -> true)
            .patch(fromOptions, reporter);
    String msg = Iterables.getOnlyElement(eventCollector.filtered(EventKind.INFO)).getMessage();

    assertThat(msg).contains("total option differences: 1");
    assertThat(msg).contains("CoreOptions stamp: trans1 mode=true, trans2 mode=false");
  }

  @Test
  public void differentDefineValues() throws Exception {
    PatchTransition trans1 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).commandLineBuildVariables =
              ImmutableList.of(Map.entry("myvar", "1"));
          return toOptions;
        };
    PatchTransition trans2 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).commandLineBuildVariables =
              ImmutableList.of(Map.entry("myvar", "2"));
          return toOptions;
        };
    BuildOptionsView fromOptions =
        new BuildOptionsView(
            targetConfig.getOptions(), targetConfig.getOptions().getFragmentClasses());

    var unused =
        new ComparingTransition(trans1, "trans1", trans2, "trans2", b -> true)
            .patch(fromOptions, reporter);
    String msg = Iterables.getOnlyElement(eventCollector.filtered(EventKind.INFO)).getMessage();

    assertThat(msg).contains("total option differences: 1");
    assertThat(msg).contains("user-defined define myvar (index 0): trans1 mode=1, trans2 mode=2");
  }

  @Test
  public void differentDefineOrder() throws Exception {
    PatchTransition trans1 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).commandLineBuildVariables =
              ImmutableList.of(Map.entry("var1", "1"), Map.entry("var2", "2"));
          return toOptions;
        };
    PatchTransition trans2 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).commandLineBuildVariables =
              ImmutableList.of(Map.entry("var2", "2"), Map.entry("var1", "1"));
          return toOptions;
        };
    BuildOptionsView fromOptions =
        new BuildOptionsView(
            targetConfig.getOptions(), targetConfig.getOptions().getFragmentClasses());

    var unused =
        new ComparingTransition(trans1, "trans1", trans2, "trans2", b -> true)
            .patch(fromOptions, reporter);
    String msg = Iterables.getOnlyElement(eventCollector.filtered(EventKind.INFO)).getMessage();

    assertThat(msg).contains("total option differences: 4");
    assertThat(msg).contains("only in trans1 mode: --user-defined define var1 (index 0)=1");
    assertThat(msg).contains("only in trans1 mode: --user-defined define var2 (index 1)=2");
    assertThat(msg).contains("only in trans2 mode: --user-defined define var2 (index 0)=2");
    assertThat(msg).contains("only in trans2 mode: --user-defined define var1 (index 1)=1");
  }

  @Test
  public void differentFeaturesValues() throws Exception {
    PatchTransition trans1 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).defaultFeatures = ImmutableList.of("a");
          return toOptions;
        };
    PatchTransition trans2 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).defaultFeatures = ImmutableList.of("a", "b");
          return toOptions;
        };
    BuildOptionsView fromOptions =
        new BuildOptionsView(
            targetConfig.getOptions(), targetConfig.getOptions().getFragmentClasses());

    var unused =
        new ComparingTransition(trans1, "trans1", trans2, "trans2", b -> true)
            .patch(fromOptions, reporter);
    String msg = Iterables.getOnlyElement(eventCollector.filtered(EventKind.INFO)).getMessage();

    assertThat(msg).contains("total option differences: 1");
    assertThat(msg).contains("only in trans2 mode: --user-defined feature b (index 1)");
  }

  @Test
  public void differentFeaturesOrder() throws Exception {
    PatchTransition trans1 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).defaultFeatures = ImmutableList.of("a", "b");
          return toOptions;
        };
    PatchTransition trans2 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).defaultFeatures = ImmutableList.of("b", "a");
          return toOptions;
        };
    BuildOptionsView fromOptions =
        new BuildOptionsView(
            targetConfig.getOptions(), targetConfig.getOptions().getFragmentClasses());

    var unused =
        new ComparingTransition(trans1, "trans1", trans2, "trans2", b -> true)
            .patch(fromOptions, reporter);
    String msg = Iterables.getOnlyElement(eventCollector.filtered(EventKind.INFO)).getMessage();

    assertThat(msg).contains("total option differences: 4");
    assertThat(msg).contains("only in trans1 mode: --user-defined feature a (index 0)");
    assertThat(msg).contains("only in trans1 mode: --user-defined feature b (index 1)");
    assertThat(msg).contains("only in trans2 mode: --user-defined feature b (index 0)");
    assertThat(msg).contains("only in trans2 mode: --user-defined feature a (index 1)");
  }

  @Test
  public void differentHostFeaturesValues() throws Exception {
    PatchTransition trans1 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).hostFeatures = ImmutableList.of("a");
          return toOptions;
        };
    PatchTransition trans2 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).hostFeatures = ImmutableList.of("a", "b");
          return toOptions;
        };
    BuildOptionsView fromOptions =
        new BuildOptionsView(
            targetConfig.getOptions(), targetConfig.getOptions().getFragmentClasses());

    var unused =
        new ComparingTransition(trans1, "trans1", trans2, "trans2", b -> true)
            .patch(fromOptions, reporter);
    String msg = Iterables.getOnlyElement(eventCollector.filtered(EventKind.INFO)).getMessage();

    assertThat(msg).contains("total option differences: 1");
    assertThat(msg).contains("only in trans2 mode: --user-defined host feature b (index 1)");
  }

  @Test
  public void differentHostFeaturesOrder() throws Exception {
    PatchTransition trans1 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).hostFeatures = ImmutableList.of("a", "b");
          return toOptions;
        };
    PatchTransition trans2 =
        (options, eventHandler) -> {
          BuildOptions toOptions = options.underlying().clone();
          toOptions.get(CoreOptions.class).hostFeatures = ImmutableList.of("b", "a");
          return toOptions;
        };
    BuildOptionsView fromOptions =
        new BuildOptionsView(
            targetConfig.getOptions(), targetConfig.getOptions().getFragmentClasses());

    var unused =
        new ComparingTransition(trans1, "trans1", trans2, "trans2", b -> true)
            .patch(fromOptions, reporter);
    String msg = Iterables.getOnlyElement(eventCollector.filtered(EventKind.INFO)).getMessage();

    assertThat(msg).contains("total option differences: 4");
    assertThat(msg).contains("only in trans1 mode: --user-defined host feature a (index 0)");
    assertThat(msg).contains("only in trans1 mode: --user-defined host feature b (index 1)");
    assertThat(msg).contains("only in trans2 mode: --user-defined host feature b (index 0)");
    assertThat(msg).contains("only in trans2 mode: --user-defined host feature a (index 1)");
  }

  @Test
  public void differentStarlarkFlagValues() throws Exception {
    PatchTransition trans1 =
        (options, eventHandler) ->
            options.underlying().toBuilder()
                .addStarlarkOption(Label.parseCanonicalUnchecked("//foo"), "1")
                .build();
    PatchTransition trans2 =
        (options, eventHandler) ->
            options.underlying().toBuilder()
                .addStarlarkOption(Label.parseCanonicalUnchecked("//foo"), "2")
                .build();
    BuildOptionsView fromOptions =
        new BuildOptionsView(
            targetConfig.getOptions(), targetConfig.getOptions().getFragmentClasses());

    var unused =
        new ComparingTransition(trans1, "trans1", trans2, "trans2", b -> true)
            .patch(fromOptions, reporter);
    String msg = Iterables.getOnlyElement(eventCollector.filtered(EventKind.INFO)).getMessage();

    assertThat(msg).contains("total option differences: 1");
    assertThat(msg).contains("--user-defined  //foo:foo (index 0): trans1 mode=1, trans2 mode=2");
  }

  @Test
  public void starlarkFlagOrderAutomaticallySorted() throws Exception {
    PatchTransition trans1 =
        (options, eventHandler) ->
            options.underlying().toBuilder()
                .addStarlarkOption(Label.parseCanonicalUnchecked("//a"), "a")
                .addStarlarkOption(Label.parseCanonicalUnchecked("//b"), "b")
                .build();
    PatchTransition trans2 =
        (options, eventHandler) ->
            options.underlying().toBuilder()
                .addStarlarkOption(Label.parseCanonicalUnchecked("//b"), "b")
                .addStarlarkOption(Label.parseCanonicalUnchecked("//a"), "a")
                .build();
    BuildOptionsView fromOptions =
        new BuildOptionsView(
            targetConfig.getOptions(), targetConfig.getOptions().getFragmentClasses());

    var unused =
        new ComparingTransition(trans1, "trans1", trans2, "trans2", b -> true)
            .patch(fromOptions, reporter);
    String msg = Iterables.getOnlyElement(eventCollector.filtered(EventKind.INFO)).getMessage();

    assertThat(msg).contains("total option differences: 0");
  }
}
