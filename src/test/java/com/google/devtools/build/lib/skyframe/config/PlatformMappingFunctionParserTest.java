// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.skyframe.config;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.Label.RepoContext;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.cmdline.RepositoryName;
import com.google.devtools.build.lib.skyframe.config.PlatformMappingFunction.Mappings;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Objects;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link PlatformMappingFunction}. */
@RunWith(JUnit4.class)
public class PlatformMappingFunctionParserTest extends AnalysisTestCase {

  private static final Label PLATFORM1 = Label.parseCanonicalUnchecked("//platforms:one");
  private static final Label PLATFORM2 = Label.parseCanonicalUnchecked("//platforms:two");
  private static final Label EXTERNAL_PLATFORM =
      Label.parseCanonicalUnchecked("@dep+1.0//platforms:two");

  @Test
  public void testParse() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "platforms:",
            "  //platforms:one",
            "    --cpu=one",
            "  //platforms:two",
            "    --cpu=two",
            "flags:",
            "  --cpu=one",
            "    //platforms:one",
            "  --cpu=two",
            "    //platforms:two");

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1, PLATFORM2);
    assertThat(mappings.platformsToFlags.get(PLATFORM1).nativeFlags()).containsExactly("--cpu=one");
    assertThat(mappings.platformsToFlags.get(PLATFORM2).nativeFlags()).containsExactly("--cpu=two");

    assertThat(mappings.flagsToPlatforms.keySet())
        .containsExactly(createFlags("--cpu=one"), createFlags("--cpu=two"));
    assertThat(mappings.flagsToPlatforms.get(createFlags("--cpu=one"))).isEqualTo(PLATFORM1);
    assertThat(mappings.flagsToPlatforms.get(createFlags("--cpu=two"))).isEqualTo(PLATFORM2);
  }

  @Test
  public void testParseWithRepoMapping() throws Exception {
    RepositoryMapping repoMapping =
        RepositoryMapping.create(
            ImmutableMap.of("foo", RepositoryName.MAIN, "dep", RepositoryName.create("dep+1.0")),
            RepositoryName.MAIN);
    PlatformMappingFunction.Mappings mappings =
        parse(
            repoMapping,
            "platforms:",
            "  @foo//platforms:one",
            "    --cpu=one",
            "  @dep//platforms:two",
            "    --cpu=two",
            "flags:",
            "  --cpu=one",
            "    @foo//platforms:one",
            "  --cpu=two",
            "    @dep//platforms:two");

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1, EXTERNAL_PLATFORM);
    assertThat(mappings.platformsToFlags.get(PLATFORM1).nativeFlags()).containsExactly("--cpu=one");
    assertThat(mappings.platformsToFlags.get(EXTERNAL_PLATFORM).nativeFlags())
        .containsExactly("--cpu=two");

    assertThat(mappings.flagsToPlatforms.keySet())
        .containsExactly(
            createFlags(repoMapping, "--cpu=one"), createFlags(repoMapping, "--cpu=two"));
    assertThat(mappings.flagsToPlatforms.get(createFlags(repoMapping, "--cpu=one")))
        .isEqualTo(PLATFORM1);
    assertThat(mappings.flagsToPlatforms.get(createFlags(repoMapping, "--cpu=two")))
        .isEqualTo(EXTERNAL_PLATFORM);
  }

  @Test
  public void testParseComment() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "# A mapping file!",
            "platforms:",
            "  # comment1",
            "  //platforms:one",
            "# comment2",
            "    --cpu=one",
            "  //platforms:two",
            "    --cpu=two",
            "flags:",
            "# another comment",
            "  --cpu=one",
            "    //platforms:one",
            "  --cpu=two",
            "    //platforms:two");

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1, PLATFORM2);
    assertThat(mappings.platformsToFlags.get(PLATFORM1).nativeFlags()).containsExactly("--cpu=one");
    assertThat(mappings.platformsToFlags.get(PLATFORM2).nativeFlags()).containsExactly("--cpu=two");

    assertThat(mappings.flagsToPlatforms.keySet())
        .containsExactly(createFlags("--cpu=one"), createFlags("--cpu=two"));
    assertThat(mappings.flagsToPlatforms.get(createFlags("--cpu=one"))).isEqualTo(PLATFORM1);
    assertThat(mappings.flagsToPlatforms.get(createFlags("--cpu=two"))).isEqualTo(PLATFORM2);
  }

  @Test
  public void testParseWhitespace() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "",
            "platforms:",
            "  ",
            "  //platforms:one",
            "",
            "    --cpu=one",
            "    //platforms:two    ",
            "      --cpu=two ",
            "flags:",
            "           ",
            "",
            "--cpu=one",
            "  //platforms:one",
            "  --cpu=two",
            "  //platforms:two");

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1, PLATFORM2);
    assertThat(mappings.platformsToFlags.get(PLATFORM1).nativeFlags()).containsExactly("--cpu=one");
    assertThat(mappings.platformsToFlags.get(PLATFORM2).nativeFlags()).containsExactly("--cpu=two");

    assertThat(mappings.flagsToPlatforms.keySet())
        .containsExactly(createFlags("--cpu=one"), createFlags("--cpu=two"));
    assertThat(mappings.flagsToPlatforms.get(createFlags("--cpu=one"))).isEqualTo(PLATFORM1);
    assertThat(mappings.flagsToPlatforms.get(createFlags("--cpu=two"))).isEqualTo(PLATFORM2);
  }

  @Test
  public void testParseMultipleFlagsInPlatform() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "platforms:",
            "  //platforms:one",
            "    --cpu=one",
            "    --compilation_mode=dbg",
            "  //platforms:two",
            "    --cpu=two");

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1, PLATFORM2);
    assertThat(mappings.platformsToFlags.get(PLATFORM1).nativeFlags())
        .containsExactly("--cpu=one", "--compilation_mode=dbg");
  }

  @Test
  public void testParseMultipleFlagsInFlags() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "flags:",
            "  --compilation_mode=dbg",
            "  --cpu=one",
            "    //platforms:one",
            "  --cpu=two",
            "    //platforms:two");

    assertThat(mappings.flagsToPlatforms.keySet())
        .containsExactly(
            createFlags("--compilation_mode=dbg", "--cpu=one"), createFlags("--cpu=two"));
    assertThat(mappings.flagsToPlatforms.get(createFlags("--compilation_mode=dbg", "--cpu=one")))
        .isEqualTo(PLATFORM1);
  }

  @Test
  public void testParseOnlyPlatforms() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "platforms:", // Force line break
            "  //platforms:one", // Force line break
            "    --cpu=one" // Force line break
            );

    assertThat(mappings.platformsToFlags.keySet()).containsExactly(PLATFORM1);
    assertThat(mappings.platformsToFlags.get(PLATFORM1).nativeFlags()).containsExactly("--cpu=one");
    assertThat(mappings.flagsToPlatforms).isEmpty();
  }

  @Test
  public void testParseOnlyFlags() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "flags:", // Force line break
            "  --cpu=one", // Force line break
            "    //platforms:one" // Force line break
            );

    assertThat(mappings.flagsToPlatforms.keySet()).containsExactly(createFlags("--cpu=one"));
    assertThat(mappings.flagsToPlatforms.get(createFlags("--cpu=one"))).isEqualTo(PLATFORM1);
    assertThat(mappings.platformsToFlags).isEmpty();
  }

  @Test
  public void testParseEmpty() throws Exception {
    PlatformMappingFunction.Mappings mappings = parse();

    assertThat(mappings.flagsToPlatforms).isEmpty();
    assertThat(mappings.platformsToFlags).isEmpty();
  }

  @Test
  public void testParseEmptySections() throws Exception {
    PlatformMappingFunction.Mappings mappings = parse("platforms:", "flags:");

    assertThat(mappings.flagsToPlatforms).isEmpty();
    assertThat(mappings.platformsToFlags).isEmpty();
  }

  @Test
  public void testParseCommentOnly() throws Exception {
    PlatformMappingFunction.Mappings mappings = parse("#No mappings");

    assertThat(mappings.flagsToPlatforms).isEmpty();
    assertThat(mappings.platformsToFlags).isEmpty();
  }

  @Test
  public void testParseExtraPlatformInFlags() throws Exception {
    PlatformMappingParsingException exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "flags:", // Force line break
                    "  --cpu=one", // Force line break
                    "    //platforms:one", // Force line break
                    "    //platforms:two" // Force line break
                    ));

    assertThat(exception).hasMessageThat().contains("//platforms:two");
  }

  @Test
  public void testParsePlatformWithoutFlags() throws Exception {
    PlatformMappingParsingException exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "platforms:", // Force line break
                    "  //platforms:one" // Force line break
                    ));

    assertThat(exception).hasMessageThat().contains("end of file");
  }

  @Test
  public void testParseFlagsWithoutPlatform() throws Exception {
    PlatformMappingParsingException exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "flags:", // Force line break
                    "  --cpu=one" // Force line break
                    ));

    assertThat(exception).hasMessageThat().contains("end of file");
  }

  @Test
  public void testParseCommentEndOfFile() throws Exception {
    PlatformMappingFunction.Mappings mappings =
        parse(
            "platforms:", // Force line break
            "  //platforms:one", // Force line break
            "    --cpu=one", // Force line break
            "# No more mappings" // Force line break
            );

    assertThat(mappings.platformsToFlags).isNotEmpty();
  }

  @Test
  public void testParseUnknownSection() throws Exception {
    PlatformMappingParsingException exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "platform:", // Force line break
                    "  //platforms:one", // Force line break
                    "    --cpu=one" // Force line break
                    ));

    assertThat(exception).hasMessageThat().contains("platform:");

    exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "platforms:",
                    "  //platforms:one",
                    "    --cpu=one",
                    "flag:",
                    "  --cpu=one",
                    "    //platforms:one"));

    assertThat(exception).hasMessageThat().contains("platform");
  }

  @Test
  public void testParsePlatformsInvalidPlatformLabel() throws Exception {
    PlatformMappingParsingException exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "platforms:", // Force line break
                    "  @@@", // Force line break
                    "    --cpu=one"));

    assertThat(exception).hasMessageThat().contains("@@@");
  }

  @Test
  public void testParseFlagsInvalidPlatformLabel() throws Exception {
    PlatformMappingParsingException exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "flags:", // Force line break
                    "  --cpu=one", // Force line break
                    "    @@@"));

    assertThat(exception).hasMessageThat().contains("@@@");
  }

  @Test
  public void testParsePlatformsInvalidFlag() throws Exception {
    PlatformMappingParsingException exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "platforms:", // Force line break
                    "  //platforms:one", // Force line break
                    "    -cpu=one"));

    assertThat(exception).hasMessageThat().contains("-cpu");
  }

  @Test
  public void testParseFlagsInvalidFlag() throws Exception {
    PlatformMappingParsingException exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "flags:", // Force line break
                    "  -cpu=one", // Force line breakPlatformMappingFunction
                    "    //platforms:one"));

    assertThat(exception).hasMessageThat().contains("-cpu");
  }

  @Test
  public void testParsePlatformsDuplicatePlatform() throws Exception {
    PlatformMappingParsingException exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "platforms:", // Force line break
                    "  //platforms:one", // Force line break
                    "    --cpu=one", // Force line break
                    "  //platforms:one", // Force line break
                    "    --cpu=two"));

    assertThat(exception).hasMessageThat().contains("duplicate");
  }

  @Test
  public void testParseFlagsDuplicateFlags() throws Exception {
    PlatformMappingParsingException exception =
        assertThrows(
            PlatformMappingParsingException.class,
            () ->
                parse(
                    "flags:", // Force line break
                    "  --compilation_mode=dbg", // Force line break
                    "  --cpu=one", // Force line break:242
                    "    //platforms:one", // Force line break
                    "  --compilation_mode=dbg", // Force line break
                    "  --cpu=one", // Force line break
                    "    //platforms:two"));

    assertThat(exception).hasMessageThat().contains("duplicate");
  }

  private NativeAndStarlarkFlags createFlags(String... nativeFlags) {
    return createFlags(RepositoryMapping.ALWAYS_FALLBACK, nativeFlags);
  }

  private NativeAndStarlarkFlags createFlags(
      RepositoryMapping mainRepoMapping, String... nativeFlags) {
    return NativeAndStarlarkFlags.builder()
        .nativeFlags(ImmutableList.copyOf(nativeFlags))
        .optionsClasses(ruleClassProvider.getFragmentRegistry().getOptionsClasses())
        .repoMapping(mainRepoMapping)
        .build();
  }

  private PlatformMappingFunction.Mappings parse(String... lines)
      throws PlatformMappingParsingException, InterruptedException {
    return parse(RepositoryMapping.ALWAYS_FALLBACK, lines);
  }

  private PlatformMappingFunction.Mappings parse(RepositoryMapping mainRepoMapping, String... lines)
      throws InterruptedException, PlatformMappingParsingException {
    Key key = Key.create(mainRepoMapping, ImmutableList.copyOf(lines));
    try {
      // Must re-enable analysis for Skyframe functions that create configured targets.
      skyframeExecutor.getSkyframeBuildView().enableAnalysis(true);
      EvaluationResult<Value> evalResult =
          SkyframeExecutorTestUtils.evaluate(
              skyframeExecutor, key, /* keepGoing= */ false, reporter);
      if (evalResult.hasError()) {
        ErrorInfo errorInfo = evalResult.getError(key);
        PlatformMappingParsingException exception =
            (PlatformMappingParsingException) errorInfo.getException();
        throw exception;
      }
      return evalResult.get(key).mappings();
    } finally {
      skyframeExecutor.getSkyframeBuildView().enableAnalysis(false);
    }
  }

  static final SkyFunctionName SKYFUNCTION_NAME = SkyFunctionName.createHermetic("PARSE_MAPPINGS");

  @AutoCodec
  static final class Key implements SkyKey {
    static Key create(RepositoryMapping mainRepoMapping, ImmutableList<String> lines) {
      return new Key(mainRepoMapping, lines);
    }

    private final RepositoryMapping mainRepoMapping;
    private final ImmutableList<String> lines;

    public Key(RepositoryMapping mainRepoMapping, ImmutableList<String> lines) {
      this.mainRepoMapping = mainRepoMapping;
      this.lines = lines;
    }

    RepositoryMapping mainRepoMapping() {
      return mainRepoMapping;
    }

    ImmutableList<String> lines() {
      return lines;
    }

    @Override
    public int hashCode() {
      return Objects.hash(mainRepoMapping, lines);
    }

    @Override
    public boolean equals(Object o) {
      if (o == null || !(o instanceof Key other)) {
        return false;
      }
      return Objects.equals(mainRepoMapping, other.mainRepoMapping)
          && Objects.equals(lines, other.lines);
    }

    @Override
    public SkyFunctionName functionName() {
      return SKYFUNCTION_NAME;
    }
  }

  static final class Value implements SkyValue {
    static Value create(Mappings mappings) {
      return new Value(mappings);
    }

    private final Mappings mappings;

    Value(Mappings mappings) {
      this.mappings = mappings;
    }

    Mappings mappings() {
      return mappings;
    }
  }

  private static final class ParseMappingsFunction implements SkyFunction {

    @Nullable
    @Override
    public Value compute(SkyKey skyKey, Environment env)
        throws InterruptedException, EvalException {
      Key key = (Key) skyKey.argument();
      try {
        Mappings mappings =
            PlatformMappingFunction.parse(
                env, key.lines(), RepoContext.of(RepositoryName.MAIN, key.mainRepoMapping()));
        if (mappings == null) {
          return null;
        }
        return Value.create(mappings);
      } catch (PlatformMappingParsingException e) {
        throw new EvalException(e);
      }
    }

    private static final class EvalException extends SkyFunctionException {
      EvalException(Exception cause) {
        super(cause, Transience.PERSISTENT); // We can generalize the transience if/when needed.
      }
    }
  }

  private static final class CustomAnalysisMock extends AnalysisMock.Delegate {

    CustomAnalysisMock() {
      super(AnalysisMock.get());
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
        BlazeDirectories directories) {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(super.getSkyFunctions(directories))
          .put(SKYFUNCTION_NAME, new ParseMappingsFunction())
          .buildOrThrow();
    }
  }

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new CustomAnalysisMock();
  }
}
