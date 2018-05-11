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
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsDiff;
import com.google.devtools.build.lib.analysis.config.BuildOptions.OptionsDiffForReconstruction;
import com.google.devtools.build.lib.rules.cpp.CppOptions;
import com.google.devtools.build.lib.rules.proto.ProtoConfiguration;
import com.google.devtools.build.lib.skyframe.serialization.testutils.ObjectCodecTester;
import com.google.devtools.common.options.OptionsParser;
import java.util.IdentityHashMap;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * A test for {@link BuildOptions}.
 */
@RunWith(JUnit4.class)
public class BuildOptionsTest {
  private static final ImmutableList<Class<? extends FragmentOptions>> TEST_OPTIONS =
      ImmutableList.<Class<? extends FragmentOptions>>of(
          BuildConfiguration.Options.class, ProtoConfiguration.Options.class);

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
    assertThat(diff.getExtraFirstFragmentClasses()).containsExactly(CppOptions.class);
    assertThat(diff.getExtraSecondFragmentClasses()).containsExactlyElementsIn(TEST_OPTIONS);
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
  public void applyDiff() throws Exception {
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
  public void testCodec() throws Exception {
    BuildOptions one =
        BuildOptions.of(
            TEST_OPTIONS,
            "--compilation_mode=opt",
            "cpu=k8",
            "--proto_compiler=//net/proto2/compiler/public:protocol_compiler",
            "--proto_toolchain_for_java=//tools/proto/toolchains:java");
    BuildOptions two =
        BuildOptions.of(
            TEST_OPTIONS,
            "--compilation_mode=dbg",
            "cpu=k8",
            "--proto_compiler=@com_google_protobuf//:protoc");
    ObjectCodecTester.newBuilder(new BuildOptions.OptionsDiffForReconstructionCodec())
        .addSubjects(BuildOptions.diffForReconstruction(one, two))
        .addDependency(
            IdentityHashMap.class,
            new IdentityHashMap<BuildOptions.OptionsDiffForReconstruction, byte[]>())
        .skipBadDataTest() // Bad data doesn't make sense with our caching.
        .verificationFunction(
            ((original, deserialized) -> {
              assertThat(original).isEqualTo(deserialized);
            }))
        .buildAndRunTests();
  }
}
