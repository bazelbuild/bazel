// Copyright 2018 The Bazel Authors. All Rights Reserved.
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
package com.google.devtools.build.lib.exec;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.devtools.build.lib.actions.ActionContext;
import com.google.devtools.build.lib.actions.ExecutorInitException;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link ModuleActionContextRegistry}. */
@RunWith(JUnit4.class)
public class ModuleActionContextRegistryTest {

  @Test
  public void testRegistration() throws Exception {
    AC2 context = new AC2();
    ModuleActionContextRegistry contextRegistry =
        ModuleActionContextRegistry.builder().register(IT1.class, context).build();
    assertThat(contextRegistry.getContext(IT1.class)).isEqualTo(context);
  }

  @Test
  public void testDoubleRegistration() throws Exception {
    AC2 context = new AC2();
    ModuleActionContextRegistry contextRegistry =
        ModuleActionContextRegistry.builder()
            .register(IT1.class, context)
            .register(IT1.class, context)
            .build();
    assertThat(contextRegistry.getContext(IT1.class)).isEqualTo(context);
  }

  @Test
  public void testLastRegisteredHasPriority() throws Exception {
    AC2 context1 = new AC2();
    AC2 context2 = new AC2();
    ModuleActionContextRegistry contextRegistry =
        ModuleActionContextRegistry.builder()
            .register(IT1.class, context1)
            .register(IT1.class, context2)
            .build();
    assertThat(contextRegistry.getContext(IT1.class)).isEqualTo(context2);
  }

  @Test
  public void testSelfIdentifyingType() throws Exception {
    AC1 context = new AC1();
    ModuleActionContextRegistry contextRegistry =
        ModuleActionContextRegistry.builder().register(AC1.class, context).build();
    assertThat(contextRegistry.getContext(AC1.class)).isEqualTo(context);
  }

  @Test
  public void testIdentifierFilter() throws Exception {
    AC2 general = new AC2();
    AC2 specific = new AC2();
    ModuleActionContextRegistry contextRegistry =
        ModuleActionContextRegistry.builder()
            .register(IT1.class, general)
            .register(IT1.class, specific, "specific", "foo")
            .register(IT1.class, general)
            .restrictTo(IT1.class, "specific")
            .build();
    assertThat(contextRegistry.getContext(IT1.class)).isEqualTo(specific);
  }

  @Test
  public void testLastRegisteredHasPriorityWithIdentifier() throws Exception {
    AC2 context1 = new AC2();
    AC2 context2 = new AC2();
    ModuleActionContextRegistry contextRegistry =
        ModuleActionContextRegistry.builder()
            .register(IT1.class, context1, "foo")
            .register(IT1.class, context2, "foo")
            .restrictTo(IT1.class, "foo")
            .build();
    assertThat(contextRegistry.getContext(IT1.class)).isEqualTo(context2);
  }

  @Test
  public void testUsedNotification() throws Exception {
    RecordingContext context = new RecordingContext();
    ModuleActionContextRegistry contextRegistry =
        ModuleActionContextRegistry.builder()
            .register(RecordingContext.class, context)
            .register(RecordingContext.class, context)
            .build();

    contextRegistry.notifyUsed();

    assertThat(context.usedCalls).isEqualTo(1);
  }

  @Test
  public void testEmptyRestriction() throws Exception {
    AC2 general = new AC2();
    AC2 specific = new AC2();
    ModuleActionContextRegistry contextRegistry =
        ModuleActionContextRegistry.builder()
            .register(IT1.class, general)
            .register(IT1.class, specific, "specific", "foo")
            .register(IT1.class, general)
            .restrictTo(IT1.class, "specific")
            .restrictTo(IT1.class, "")
            .build();
    assertThat(contextRegistry.getContext(IT1.class)).isEqualTo(general);
  }

  @Test
  public void testNoMatch() throws Exception {
    ModuleActionContextRegistry contextRegistry =
        ModuleActionContextRegistry.builder().register(AC1.class, new AC1()).build();

    assertThat(contextRegistry.getContext(IT1.class)).isNull();
  }

  @Test
  public void testUnfulfilledRestriction() {
    AC2 context1 = new AC2();
    AC2 context2 = new AC2();
    ModuleActionContextRegistry.Builder builder =
        ModuleActionContextRegistry.builder()
            .register(IT1.class, context1, "foo")
            .register(IT1.class, context2, "baz", "boz")
            .restrictTo(IT1.class, "bar");

    ExecutorInitException exception = assertThrows(ExecutorInitException.class, builder::build);
    assertThat(exception).hasMessageThat().containsMatch("IT1.*bar.*[foo, baz, boz]");
  }

  private static class AC1 implements ActionContext {}

  private interface IT1 extends ActionContext {}

  private static class AC2 implements IT1 {}

  private static class RecordingContext implements ActionContext {
    private int usedCalls = 0;

    @Override
    public void usedContext(ActionContext.ActionContextRegistry actionContextRegistry) {
      usedCalls++;
    }
  }
}
