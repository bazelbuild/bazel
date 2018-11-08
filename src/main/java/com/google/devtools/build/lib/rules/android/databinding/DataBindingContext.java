// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.rules.android.databinding;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.RuleConfiguredTargetBuilder;
import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.rules.android.AndroidResources;
import com.google.devtools.build.lib.rules.java.JavaPluginInfoProvider;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

/** Contains Android Databinding configuration and resource generation information. */
public interface DataBindingContext {

  /**
   * Returns the file where data binding's resource processing produces binding xml. For example,
   * given:
   *
   * <pre>{@code
   * <layout>
   *   <data>
   *     <variable name="foo" type="String" />
   *   </data>
   * </layout>
   * <LinearLayout>
   *   ...
   * </LinearLayout>
   * }</pre>
   *
   * <p>data binding strips out and processes this part:
   *
   * <pre>{@code
   * <data>
   *   <variable name="foo" type="String" />
   * </data>
   * }</pre>
   *
   * for each layout file with data binding expressions. Since this may produce multiple files,
   * outputs are zipped up into a single container.
   */
  void supplyLayoutInfo(Consumer<Artifact> consumer);

  /** The javac flags that are needed to configure data binding's annotation processor. */
  void supplyJavaCoptsUsing(
      RuleContext ruleContext, boolean isBinary, Consumer<Iterable<String>> consumer);

  /**
   * Adds data binding's annotation processor as a plugin to the given Java compilation context.
   *
   * <p>This extends the Java compilation to translate data binding .xml into corresponding
   * classes.
   */
  void supplyAnnotationProcessor(
      RuleContext ruleContext, BiConsumer<JavaPluginInfoProvider, Iterable<Artifact>> consumer);

  /**
   * Processes deps that also apply data binding.
   *
   * @param ruleContext the current rule
   * @return the deps' metadata outputs. These need to be staged as compilation inputs to the
   *     current rule.
   */
  ImmutableList<Artifact> processDeps(RuleContext ruleContext);

  /**
   * Creates and adds the generated Java source for data binding annotation processor to read and
   * translate layout info xml (from {@link #supplyLayoutInfo(Consumer)} into the classes that end
   * user code consumes.
   *
   * <p>This triggers the annotation processor. Annotation processor settings are configured
   * separately in {@link #supplyJavaCoptsUsing(RuleContext, boolean, Consumer)}.
   */
  ImmutableList<Artifact> addAnnotationFileToSrcs(
      ImmutableList<Artifact> srcs, RuleContext ruleContext);

  /**
   * Adds the appropriate {@link UsesDataBindingProvider} for a rule if it should expose one.
   *
   * <p>A rule exposes {@link UsesDataBindingProvider} if either it or its deps set {@code
   * enable_data_binding = 1}.
   */
  void addProvider(RuleConfiguredTargetBuilder builder, RuleContext ruleContext);

  AndroidResources processResources(AndroidResources resources);
}
