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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCacheReaderDepsProvider;
import com.google.devtools.build.lib.skyframe.serialization.analysis.RemoteAnalysisCachingOptions.RemoteAnalysisCacheMode;
import com.google.devtools.build.skyframe.AbstractSkyFunctionEnvironmentForTesting;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.ValueOrUntypedException;
import java.util.HashMap;
import java.util.Map;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ActionLookupConflictFindingFunction}. */
@RunWith(JUnit4.class)
public final class ActionLookupConflictFindingFunctionTest {

  private static final class TestEnvironment extends AbstractSkyFunctionEnvironmentForTesting {
    private final Map<SkyKey, SkyValue> values = new HashMap<>();

    @Override
    protected ImmutableMap<SkyKey, ValueOrUntypedException> getValueOrUntypedExceptions(
        Iterable<? extends SkyKey> depKeys) {
      ImmutableMap.Builder<SkyKey, ValueOrUntypedException> builder = ImmutableMap.builder();
      for (SkyKey key : depKeys) {
        SkyValue val = values.get(key);
        if (val == null) {
          this.valuesMissing = true;
        }
        builder.put(key, ValueOrUntypedException.ofValueUntyped(val));
      }
      return builder.buildOrThrow();
    }

    @Override
    public ExtendedEventHandler getListener() {
      return mock(ExtendedEventHandler.class);
    }
  }

  @Test
  public void compute_missingValue_isRetrievalEnabled_noBugReport() throws Exception {
    RemoteAnalysisCacheReaderDepsProvider provider =
        mock(RemoteAnalysisCacheReaderDepsProvider.class);
    when(provider.mode()).thenReturn(RemoteAnalysisCacheMode.DOWNLOAD);

    ActionLookupConflictFindingFunction function =
        new ActionLookupConflictFindingFunction(() -> provider);

    ActionLookupKey lookupKey =
        ConfiguredTargetKey.builder().setLabel(Label.parseCanonicalUnchecked("//foo:foo")).build();
    SkyKey key = ActionLookupConflictFindingValue.key(lookupKey);

    TestEnvironment env = new TestEnvironment();
    // ActionLookupConflictFindingFunction calls ACTION_CONFLICTS.get(env)
    env.values.put(
        ArtifactConflictFinder.ACTION_CONFLICTS.getKey(), new PrecomputedValue(ImmutableMap.of()));

    SkyValue result = function.compute(key, env);

    assertThat(result).isNull();
    assertThat(env.valuesMissing()).isTrue();
    // No exception thrown means no bug report sent.
  }

  @Test
  public void compute_missingValue_isRetrievalDisabled_bugReport() throws Exception {
    RemoteAnalysisCacheReaderDepsProvider provider =
        mock(RemoteAnalysisCacheReaderDepsProvider.class);
    when(provider.mode()).thenReturn(RemoteAnalysisCacheMode.OFF);

    ActionLookupConflictFindingFunction function =
        new ActionLookupConflictFindingFunction(() -> provider);

    ActionLookupKey lookupKey =
        ConfiguredTargetKey.builder().setLabel(Label.parseCanonicalUnchecked("//bar:bar")).build();
    SkyKey key = ActionLookupConflictFindingValue.key(lookupKey);

    TestEnvironment env = new TestEnvironment();
    env.values.put(
        ArtifactConflictFinder.ACTION_CONFLICTS.getKey(), new PrecomputedValue(ImmutableMap.of()));

    // BugReport.sendNonFatalBugReport throws IllegalStateException in tests.
    var thrown = assertThrows(IllegalStateException.class, () -> function.compute(key, env));
    assertThat(thrown)
        .hasMessageThat()
        .contains("Unexpected missing action lookup value during action conflict finding");
  }
}
