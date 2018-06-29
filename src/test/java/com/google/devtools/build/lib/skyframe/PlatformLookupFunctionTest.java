package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.PlatformLookupFunction.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link PlatformLookupFunction}. */
@RunWith(JUnit4.class)
public class PlatformLookupFunctionTest extends ToolchainTestCase {

  @Test
  public void testPlatformLookup() throws Exception {
    ConfiguredTargetKey linuxKey =
        ConfiguredTargetKey.of(makeLabel("//platforms:linux"), targetConfigKey, false);
    ConfiguredTargetKey macKey =
        ConfiguredTargetKey.of(makeLabel("//platforms:mac"), targetConfigKey, false);
    PlatformLookupValue.Key key = PlatformLookupValue.key(ImmutableList.of(linuxKey, macKey));

    EvaluationResult<PlatformLookupValue> result = requestPlatformsFromSkyframe(key);

    assertThatEvaluationResult(result).hasNoError();
    assertThatEvaluationResult(result).hasEntryThat(key).isNotNull();

    Map<ConfiguredTargetKey, PlatformInfo> platforms = result.get(key).platforms();
    assertThat(platforms).containsEntry(linuxKey, linuxPlatform);
    assertThat(platforms).containsEntry(macKey, macPlatform);
    assertThat(platforms).hasSize(2);
  }

  @Test
  public void testPlatformLookup_targetNotPlatform() throws Exception {
    scratch.file("invalid/BUILD", "filegroup(name = 'not_a_platform')");

    ConfiguredTargetKey targetKey =
        ConfiguredTargetKey.of(makeLabel("//invalid:not_a_platform"), targetConfigKey, false);
    PlatformLookupValue.Key key = PlatformLookupValue.key(ImmutableList.of(targetKey));

    EvaluationResult<PlatformLookupValue> result = requestPlatformsFromSkyframe(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//invalid:not_a_platform");
  }

  @Test
  public void testPlatformLookup_targetDoesNotExist() throws Exception {
    ConfiguredTargetKey targetKey =
        ConfiguredTargetKey.of(makeLabel("//fake:missing"), targetConfigKey, false);
    PlatformLookupValue.Key key = PlatformLookupValue.key(ImmutableList.of(targetKey));

    EvaluationResult<PlatformLookupValue> result = requestPlatformsFromSkyframe(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//fake:missing");
  }

  protected EvaluationResult<PlatformLookupValue> requestPlatformsFromSkyframe(SkyKey key)
      throws InterruptedException {
    try {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          getSkyframeExecutor(), key, /*keepGoing=*/ false, reporter);
    } finally {
      getSkyframeExecutor().getSkyframeBuildView().enableAnalysis(false);
    }
  }
}
