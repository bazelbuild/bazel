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
package com.google.devtools.build.lib.rules.cpp;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.cpp.transitions.DisableLipoTransition;
import com.google.devtools.build.lib.rules.cpp.transitions.LipoContextCollectorTransition;
import com.google.devtools.build.lib.view.config.crosstool.CrosstoolConfig.LipoMode;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests LIPO-related configuration transitions.
 **/
@RunWith(JUnit4.class)
public class LipoTransitionsTest extends BuildViewTestCase {

  private void useLipoOptimizationConfig() throws Exception {
    useConfiguration(
        "--compilation_mode=opt",
        "--fdo_optimize=profile.zip",
        "--lipo_context=//foo",
        "--lipo=binary");
  }

  private CppOptions doTransition(PatchTransition transition, BuildOptions fromOptions) {
    return transition.apply(fromOptions).get(CppOptions.class);
  }

  @Test
  public void expectedTargetConfig() throws Exception {
    useLipoOptimizationConfig();
    CppOptions targetOptions = getTargetConfiguration().getOptions().get(CppOptions.class);
    assertThat(targetOptions.isFdo()).isTrue();
    assertThat(targetOptions.isLipoOptimization()).isTrue();
    assertThat(targetOptions.isLipoOptimizationOrInstrumentation()).isTrue();
    assertThat(targetOptions.isLipoContextCollector()).isFalse();
    assertThat(targetOptions.getLipoContext()).isEqualTo(Label.parseAbsoluteUnchecked("//foo"));
    assertThat(targetOptions.getLipoContextForBuild())
        .isEqualTo(Label.parseAbsoluteUnchecked("//foo"));
    assertThat(targetOptions.getLipoMode()).isEqualTo(LipoMode.BINARY);
  }

  @Test
  public void disableLipoFromTargetConfig() throws Exception {
    useLipoOptimizationConfig();
    CppOptions toOptions =
        doTransition(DisableLipoTransition.INSTANCE, getTargetConfiguration().getOptions());
    assertThat(toOptions).isNotEqualTo(getTargetConfiguration().getOptions().get(CppOptions.class));
    assertThat(toOptions.isFdo()).isFalse();
    assertThat(toOptions.isLipoOptimization()).isFalse();
    assertThat(toOptions.isLipoOptimizationOrInstrumentation()).isFalse();
    assertThat(toOptions.isLipoContextCollector()).isFalse();
    assertThat(toOptions.getLipoContext()).isNull();
    assertThat(toOptions.getLipoContextForBuild()).isEqualTo(Label.parseAbsoluteUnchecked("//foo"));
    assertThat(toOptions.getLipoMode()).isEqualTo(LipoMode.OFF);
  }

  @Test
  public void disableLipoFromContextCollectorConfig() throws Exception {
    useLipoOptimizationConfig();
    BuildOptions contextCollectorOptions =
        LipoContextCollectorTransition.INSTANCE.apply(getTargetConfiguration().getOptions());
    CppOptions toOptions = doTransition(DisableLipoTransition.INSTANCE, contextCollectorOptions);
    assertThat(toOptions).isEqualTo(contextCollectorOptions.get(CppOptions.class));
  }

  @Test
  public void disableLipoFromAlreadyDisabledConfig() throws Exception {
    useLipoOptimizationConfig();
    BuildOptions dataOptions =
        DisableLipoTransition.INSTANCE.apply(getTargetConfiguration().getOptions());
    CppOptions toOptions = doTransition(DisableLipoTransition.INSTANCE, dataOptions);
    assertThat(toOptions).isEqualTo(dataOptions.get(CppOptions.class));
  }

  @Test
  public void disableLipoFromHostConfig() throws Exception {
    useLipoOptimizationConfig();
    CppOptions toOptions =
        doTransition(DisableLipoTransition.INSTANCE, getHostConfiguration().getOptions());
    assertThat(toOptions).isEqualTo(getHostConfiguration().getOptions().get(CppOptions.class));
  }

  @Test
  public void disableLipoNoFdoBuild() throws Exception {
    useConfiguration();
    CppOptions toOptions =
        doTransition(DisableLipoTransition.INSTANCE, getTargetConfiguration().getOptions());
    assertThat(toOptions).isEqualTo(getTargetConfiguration().getOptions().get(CppOptions.class));
  }

  @Test
  public void disableLipoFdoInstrumentBuild() throws Exception {
    useConfiguration("--fdo_instrument=profile.zip");
    CppOptions toOptions =
        doTransition(DisableLipoTransition.INSTANCE, getTargetConfiguration().getOptions());
    assertThat(toOptions).isEqualTo(getTargetConfiguration().getOptions().get(CppOptions.class));
  }

  @Test
  public void disableLipoFdoOptimizeBuild() throws Exception {
    useConfiguration("--fdo_optimize=profile.zip");
    CppOptions toOptions =
        doTransition(DisableLipoTransition.INSTANCE, getTargetConfiguration().getOptions());
    assertThat(toOptions).isEqualTo(getTargetConfiguration().getOptions().get(CppOptions.class));
  }

  @Test
  public void disableLipoLipoInstrumentBuild() throws Exception {
    useConfiguration("--fdo_instrument=profile.zip", "--lipo=binary", "--compilation_mode=opt");
    CppOptions toOptions =
        doTransition(DisableLipoTransition.INSTANCE, getTargetConfiguration().getOptions());
    assertThat(toOptions).isNotEqualTo(getTargetConfiguration().getOptions().get(CppOptions.class));
    assertThat(toOptions.isFdo()).isFalse();
    assertThat(toOptions.isLipoOptimization()).isFalse();
    assertThat(toOptions.isLipoOptimizationOrInstrumentation()).isFalse();
    assertThat(toOptions.isLipoContextCollector()).isFalse();
    assertThat(toOptions.getLipoContext()).isNull();
    assertThat(toOptions.getLipoContextForBuild()).isNull();
    assertThat(toOptions.getLipoMode()).isEqualTo(LipoMode.OFF);
  }

  @Test
  public void contextCollectorFromTargetConfig() throws Exception {
    useLipoOptimizationConfig();
    CppOptions toOptions = doTransition(LipoContextCollectorTransition.INSTANCE,
        getTargetConfiguration().getOptions());
    assertThat(toOptions).isNotEqualTo(getTargetConfiguration().getOptions().get(CppOptions.class));
    assertThat(toOptions.isFdo()).isTrue();
    assertThat(toOptions.isLipoOptimization()).isFalse();
    assertThat(toOptions.isLipoOptimizationOrInstrumentation()).isFalse();
    assertThat(toOptions.isLipoContextCollector()).isTrue();
    assertThat(toOptions.getLipoContext()).isNull();
    assertThat(toOptions.getLipoContextForBuild()).isEqualTo(Label.parseAbsoluteUnchecked("//foo"));
    assertThat(toOptions.getLipoMode()).isEqualTo(LipoMode.BINARY);
  }

  @Test
  public void contextCollectorFromContextCollectorConfig() throws Exception {
    useLipoOptimizationConfig();
    BuildOptions contextCollectorOptions =
        LipoContextCollectorTransition.INSTANCE.apply(getTargetConfiguration().getOptions());
    CppOptions toOptions =
        doTransition(LipoContextCollectorTransition.INSTANCE, contextCollectorOptions);
    assertThat(toOptions).isEqualTo(contextCollectorOptions.get(CppOptions.class));
  }

  @Test
  public void contextCollectorFromDataConfig() throws Exception {
    useLipoOptimizationConfig();
    BuildOptions dataOptions =
        DisableLipoTransition.INSTANCE.apply(getTargetConfiguration().getOptions());
    CppOptions toOptions = doTransition(LipoContextCollectorTransition.INSTANCE, dataOptions);
    assertThat(toOptions).isEqualTo(dataOptions.get(CppOptions.class));
  }

  @Test
  public void contextCollectorFromHostConfig() throws Exception {
    useLipoOptimizationConfig();
    CppOptions toOptions =
        doTransition(LipoContextCollectorTransition.INSTANCE, getHostConfiguration().getOptions());
    assertThat(toOptions).isEqualTo(getHostConfiguration().getOptions().get(CppOptions.class));
  }

  @Test
  public void contextCollectorNoFdoBuild() throws Exception {
    useConfiguration();
    CppOptions toOptions = doTransition(LipoContextCollectorTransition.INSTANCE,
        getTargetConfiguration().getOptions());
    assertThat(toOptions).isEqualTo(getTargetConfiguration().getOptions().get(CppOptions.class));
  }

  @Test
  public void contextCollectorFdoInstrumentBuild() throws Exception {
    useConfiguration("--fdo_instrument=profile.zip");
    CppOptions toOptions = doTransition(LipoContextCollectorTransition.INSTANCE,
        getTargetConfiguration().getOptions());
    assertThat(toOptions).isEqualTo(getTargetConfiguration().getOptions().get(CppOptions.class));
  }

  @Test
  public void contextCollectorFdoOptimizeBuild() throws Exception {
    useConfiguration("--fdo_optimize=profile.zip");
    CppOptions toOptions = doTransition(LipoContextCollectorTransition.INSTANCE,
        getTargetConfiguration().getOptions());
    assertThat(toOptions).isEqualTo(getTargetConfiguration().getOptions().get(CppOptions.class));
  }

  @Test
  public void contextCollectorLipoInstrumentBuild() throws Exception {
    useConfiguration("--fdo_instrument=profile.zip", "--lipo=binary", "--compilation_mode=opt");
    CppOptions toOptions = doTransition(LipoContextCollectorTransition.INSTANCE,
        getTargetConfiguration().getOptions());
    assertThat(toOptions).isEqualTo(getTargetConfiguration().getOptions().get(CppOptions.class));
  }
}
