// Copyright 2024 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.bazel.rules;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.packages.AutoloadSymbols;
import com.google.devtools.build.lib.skyframe.PrecomputedValue;
import com.google.devtools.build.skyframe.EvaluationContext;
import com.google.devtools.build.skyframe.SkyKey;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class AutoloadSymbolsTest extends BuildViewTestCase {

  private static final SkyKey AUTOLOAD_SYMBOLS_KEY = AutoloadSymbols.AUTOLOAD_SYMBOLS.getKey();
  private static final ImmutableList<SkyKey> KEYS_TO_EVALUATE =
      ImmutableList.of(AUTOLOAD_SYMBOLS_KEY);

  @Test
  public void bzlmodFlagUpdatesAutoloadConfig() throws Exception {
    EvaluationContext context =
        EvaluationContext.newBuilder().setParallelism(1).setEventHandler(reporter).build();

    setBuildLanguageOptions("--enable_bzlmod");
    AutoloadSymbols value1 = evaluateAutoloads(context);
    setBuildLanguageOptions("--noenable_bzlmod");
    AutoloadSymbols value2 = evaluateAutoloads(context);

    assertThat(value1).isNotSameInstanceAs(value2);
  }

  private AutoloadSymbols evaluateAutoloads(EvaluationContext context) throws InterruptedException {
    return (AutoloadSymbols)
        ((PrecomputedValue)
                skyframeExecutor
                    .getEvaluator()
                    .evaluate(KEYS_TO_EVALUATE, context)
                    .get(AUTOLOAD_SYMBOLS_KEY))
            .get();
  }
}
