// Copyright 2009 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.analysis.config;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration.Options;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsDiff;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsDiffForReconstruction;
import com.google.devtools.build.lib.rules.android.AndroidConfiguration;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.java.JavaOptions;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.rules.python.PythonOptions;
import com.google.devtools.build.lib.skyframe.serialization.testutils.TestUtils;
import com.google.devtools.common.options.OptionsParser;
import java.util.AbstractMap;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link BuildOptions}.
 *
 * Currently this tests native options and skylark options completely
 * separately since these two types of options do not interact. In the future when we begin to
 * migrate native options to skylark options, the format of this test class will need to accommodate
 * that overlap.
 */
@RunWith(JUnit4.class)
public class BuildOptionsTest {
  private static final ImmutableList<Class<? extends FragmentOptions>> TEST_OPTIONS =
      ImmutableList.of(BuildConfiguration.Options.class);

  @Test
  public void optionSetCaching() {
    BuildOptions a = BuildOptions.of(TEST_OPTIONS, OptionsParser.newOptionsParser(TEST_OPTIONS));
    BuildOptions b = BuildOptions.of(TEST_OPTIONS, OptionsParser.newOptionsParser(TEST_OPTIONS));
    // The cache keys of the OptionSets must be equal even if these are
    // different objects, if they were created with the same options (no options in this case).
    assertThat(b.toString()).isEqualTo(a.toString());
    assertThat(b.computeCacheKey()).isEqualTo(a.computeCacheKey());
  }

  @Test
  public void cachingSpecialCases() throws Exception {
    // You can add options here to test that their string representations are good.
    String[] options = new String[] { "--run_under=//run_under" };
    BuildOptions a = BuildOptions.of(TEST_OPTIONS, options);
    BuildOptions b = BuildOptions.of(TEST_OPTIONS, options);
    assertThat(b.toString()).isEqualTo(a.toString());
  }

  @Test
  public void optionsEquality() throws Exception {
    String[] options1 = new String[] { "--compilation_mode=opt" };
    String[] options2 = new String[] { "--compilation_mode=dbg" };
    // Distinct instances with the same values are equal:
    assertThat(BuildOptions.of(TEST_OPTIONS, options1))
        .isEqualTo(BuildOptions.of(TEST_OPTIONS, options1));
    // Same fragments, different values aren't equal:
    assertThat(
            BuildOptions.of(TEST_OPTIONS, options1).equals(BuildOptions.of(TEST_OPTIONS, options2)))
        .isFalse();
    // Same values, different fragments aren't equal:
    assertThat(
            BuildOptions.of(TEST_OPTIONS, options1)
                .equals(
                    BuildOptions.of(
                        ImmutableList.<Class<? extends FragmentOptions>>of(
                            BuildConfiguration.Options.class, CppOptions.class),
                        options1)))
        .isFalse();
  }

  @Test
  public void optionsDiff() throws Exception {
    BuildOptions one = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=opt", "cpu=k8");
    BuildOptions two = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=dbg", "cpu=k8");
    BuildOptions three = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=dbg", "cpu=k8");

    OptionsDiff diffOneTwo = BuildOptions.diff(one, two);
    OptionsDiff diffTwoThree = BuildOptions.diff(two, three);

    assertThat(diffOneTwo.areSame()).isFalse();
    assertThat(diffOneTwo.getFirst().keySet()).isEqualTo(diffOneTwo.getSecond().keySet());
    assertThat(diffOneTwo.prettyPrint()).contains("opt");
    assertThat(diffOneTwo.prettyPrint()).contains("dbg");

    assertThat(diffTwoThree.areSame()).isTrue();
  }

  @Test
  public void optionsDiff_differentFragments() throws Exception {
    BuildOptions one =
        BuildOptions.of(ImmutableList.<Class<? extends FragmentOptions>>of(CppOptions.class));
    BuildOptions two = BuildOptions.of(TEST_OPTIONS);

    OptionsDiff diff = BuildOptions.diff(one, two);

    assertThat(diff.areSame()).isFalse();
    assertThat(diff.getExtraFirstFragmentClassesForTesting()).containsExactly(CppOptions.class);
    assertThat(
            diff.getExtraSecondFragmentsForTesting()
                .stream()
                .map(Object::getClass)
                .collect(Collectors.toSet()))
        .containsExactlyElementsIn(TEST_OPTIONS);
  }

  @Test
  public void optionsDiff_nullOptionsThrow() throws Exception {
    BuildOptions one = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=opt", "cpu=k8");
    BuildOptions two = null;
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> BuildOptions.diff(one, two));
    assertThat(e).hasMessageThat().contains("Cannot diff null BuildOptions");
  }

  @Test
  public void optionsDiff_nullSecondValue() throws Exception {
    BuildOptions one = BuildOptions.of(ImmutableList.of(CppOptions.class), "--compiler=gcc");
    BuildOptions two = BuildOptions.of(ImmutableList.of(CppOptions.class));
    OptionsDiffForReconstruction diffForReconstruction =
        BuildOptions.diffForReconstruction(one, two);
    OptionsDiff diff = BuildOptions.diff(one, two);
    assertThat(diff.areSame()).isFalse();
    assertThat(diff.getSecond().values()).contains(null);
    BuildOptions reconstructed = one.applyDiff(diffForReconstruction);
    assertThat(reconstructed.get(CppOptions.class).cppCompiler).isNull();
  }

  @Test
  public void optionsDiff_differentBaseThrowException() throws Exception {
    BuildOptions one = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=opt", "cpu=k8");
    BuildOptions two = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=dbg", "cpu=k8");
    BuildOptions three = BuildOptions.of(ImmutableList.of(CppOptions.class), "--compiler=gcc");
    OptionsDiffForReconstruction diffForReconstruction =
        BuildOptions.diffForReconstruction(one, two);
    IllegalArgumentException e =
        assertThrows(IllegalArgumentException.class, () -> three.applyDiff(diffForReconstruction));
    assertThat(e)
        .hasMessageThat()
        .contains("Can not reconstruct BuildOptions with a different base");
  }

  @Test
  public void optionsDiff_getEmptyAndApplyEmpty() throws Exception {
    BuildOptions one = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=opt", "cpu=k8");
    BuildOptions two = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=opt", "cpu=k8");
    OptionsDiffForReconstruction diffForReconstruction =
        BuildOptions.diffForReconstruction(one, two);
    BuildOptions reconstructed = one.applyDiff(diffForReconstruction);
    assertThat(one).isEqualTo(reconstructed);
  }

  @Test
  public void applyDiff_nativeOptions() throws Exception {
    BuildOptions one = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=opt", "cpu=k8");
    BuildOptions two = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=dbg", "cpu=k8");
    BuildOptions reconstructedTwo = one.applyDiff(BuildOptions.diffForReconstruction(one, two));
    assertThat(reconstructedTwo).isEqualTo(two);
    assertThat(reconstructedTwo).isNotSameAs(two);
    BuildOptions reconstructedOne = one.applyDiff(BuildOptions.diffForReconstruction(one, one));
    assertThat(reconstructedOne).isSameAs(one);
    BuildOptions otherFragment = BuildOptions.of(ImmutableList.of(CppOptions.class));
    assertThat(one.applyDiff(BuildOptions.diffForReconstruction(one, otherFragment)))
        .isEqualTo(otherFragment);
    assertThat(otherFragment.applyDiff(BuildOptions.diffForReconstruction(otherFragment, one)))
        .isEqualTo(one);
  }

  @Test
  public void optionsDiff_sameStarlarkOptions() throws Exception {
    String flagName = "//foo/flag";
    String flagValue = "value";
    BuildOptions one = BuildOptions.of(ImmutableMap.of(flagName, flagValue));
    BuildOptions two = BuildOptions.of(ImmutableMap.of(flagName, flagValue));

    assertThat(BuildOptions.diff(one, two).areSame()).isTrue();
  }

  @Test
  public void optionsDiff_differentStarlarkOptions() throws Exception {
    String flagName = "//bar/flag";
    String flagValueOne = "valueOne";
    String flagValueTwo = "valueTwo";
    BuildOptions one = BuildOptions.of(ImmutableMap.of(flagName, flagValueOne));
    BuildOptions two = BuildOptions.of(ImmutableMap.of(flagName, flagValueTwo));

    OptionsDiff diff = BuildOptions.diff(one, two);

    assertThat(diff.areSame()).isFalse();
    assertThat(diff.getStarlarkFirstForTesting().keySet())
        .isEqualTo(diff.getStarlarkSecondForTesting().keySet());
    assertThat(diff.getStarlarkFirstForTesting().keySet()).containsExactly(flagName);
    assertThat(diff.getStarlarkFirstForTesting().values()).containsExactly(flagValueOne);
    assertThat(diff.getStarlarkSecondForTesting().values()).containsExactly(flagValueTwo);
  }

  @Test
  public void optionsDiff_extraStarlarkOptions() throws Exception {
    String flagNameOne = "//extra/flag/one";
    String flagNameTwo = "//extra/flag/two";
    String flagValue = "foo";
    BuildOptions one = BuildOptions.of(ImmutableMap.of(flagNameOne, flagValue));
    BuildOptions two = BuildOptions.of(ImmutableMap.of(flagNameTwo, flagValue));

    OptionsDiff diff = BuildOptions.diff(one, two);

    assertThat(diff.areSame()).isFalse();
    assertThat(diff.getExtraStarlarkOptionsFirstForTesting()).containsExactly(flagNameOne);
    assertThat(diff.getExtraStarlarkOptionsSecondForTesting().entrySet())
        .containsExactly(Maps.immutableEntry(flagNameTwo, flagValue));
  }

  @Test
  public void applyDiff_sameStarlarkOptions() throws Exception {
    String flagName = "//foo/flag";
    String flagValue = "value";
    BuildOptions one = BuildOptions.of(ImmutableMap.of(flagName, flagValue));
    BuildOptions two = BuildOptions.of(ImmutableMap.of(flagName, flagValue));

    BuildOptions reconstructedTwo = one.applyDiff(BuildOptions.diffForReconstruction(one, two));

    assertThat(reconstructedTwo).isEqualTo(two);
    assertThat(reconstructedTwo).isNotSameAs(two);

    BuildOptions reconstructedOne = one.applyDiff(BuildOptions.diffForReconstruction(one, one));

    assertThat(reconstructedOne).isSameAs(one);
  }

  @Test
  public void applyDiff_differentStarlarkOptions() throws Exception {
    String flagName = "//bar/flag";
    String flagValueOne = "valueOne";
    String flagValueTwo = "valueTwo";
    BuildOptions one = BuildOptions.of(ImmutableMap.of(flagName, flagValueOne));
    BuildOptions two = BuildOptions.of(ImmutableMap.of(flagName, flagValueTwo));

    BuildOptions reconstructedTwo = one.applyDiff(BuildOptions.diffForReconstruction(one, two));

    assertThat(reconstructedTwo).isEqualTo(two);
    assertThat(reconstructedTwo).isNotSameAs(two);
  }

  @Test
  public void applyDiff_extraStarlarkOptions() throws Exception {
    String flagNameOne = "//extra/flag/one";
    String flagNameTwo = "//extra/flag/two";
    String flagValue = "foo";
    BuildOptions one = BuildOptions.of(ImmutableMap.of(flagNameOne, flagValue));
    BuildOptions two = BuildOptions.of(ImmutableMap.of(flagNameTwo, flagValue));

    BuildOptions reconstructedTwo = one.applyDiff(BuildOptions.diffForReconstruction(one, two));
    
    assertThat(reconstructedTwo).isEqualTo(two);
    assertThat(reconstructedTwo).isNotSameAs(two);
  }

  private static ImmutableList.Builder<Class<? extends FragmentOptions>> makeOptionsClassBuilder() {
    return ImmutableList.<Class<? extends FragmentOptions>>builder()
        .addAll(TEST_OPTIONS)
        .add(CppOptions.class);
  }

  /**
   * Tests that an {@link OptionsDiffForReconstruction} serializes stably. Unfortunately, still
   * passes without fixes! (Perhaps more classes and diffs are needed?)
   */
  @Test
  public void codecStability() throws Exception {
    BuildOptions one =
        BuildOptions.of(
            makeOptionsClassBuilder()
                .add(AndroidConfiguration.Options.class)
                .add(ProtoConfiguration.Options.class)
                .build());
    BuildOptions two =
        BuildOptions.of(
            makeOptionsClassBuilder().add(JavaOptions.class).add(PythonOptions.class).build(),
            "--cpu=k8",
            "--compilation_mode=opt",
            "--compiler=gcc",
            "--copt=-Dfoo",
            "--javacopt=--javacoption",
            "--experimental_fix_deps_tool=fake",
            "--build_python_zip=no",
            "--force_python=py2");
    OptionsDiffForReconstruction diff1 = BuildOptions.diffForReconstruction(one, two);
    OptionsDiffForReconstruction diff2 = BuildOptions.diffForReconstruction(one, two);
    assertThat(diff2).isEqualTo(diff1);
    assertThat(
            TestUtils.toBytes(
                diff2,
                ImmutableMap.of(
                    BuildOptions.OptionsDiffCache.class, new BuildOptions.DiffToByteCache())))
        .isEqualTo(
            TestUtils.toBytes(
                diff1,
                ImmutableMap.of(
                    BuildOptions.OptionsDiffCache.class, new BuildOptions.DiffToByteCache())));
  }

  @Test
  public void repeatedCodec() throws Exception {
    BuildOptions one = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=opt", "cpu=k8");
    BuildOptions two = BuildOptions.of(TEST_OPTIONS, "--compilation_mode=dbg", "cpu=k8");
    OptionsDiffForReconstruction diff = BuildOptions.diffForReconstruction(one, two);
    BuildOptions.OptionsDiffCache cache = new BuildOptions.FingerprintingKDiffToByteStringCache();
    assertThat(TestUtils.toBytes(diff, ImmutableMap.of(BuildOptions.OptionsDiffCache.class, cache)))
        .isEqualTo(
            TestUtils.toBytes(diff, ImmutableMap.of(BuildOptions.OptionsDiffCache.class, cache)));
  }

  @Test
  public void testMultiValueOptionImmutability() throws Exception {
    BuildOptions options =
        BuildOptions.of(TEST_OPTIONS, OptionsParser.newOptionsParser(TEST_OPTIONS));
    BuildConfiguration.Options coreOptions = options.get(Options.class);
    assertThrows(
        UnsupportedOperationException.class,
        () ->
            coreOptions.commandLineBuildVariables.add(new AbstractMap.SimpleEntry<>("foo", "bar")));
  }
}
