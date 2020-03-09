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

package com.google.devtools.build.lib.analysis.test;

import com.google.devtools.build.lib.analysis.RuleContext;
import com.google.devtools.build.lib.analysis.platform.ConstraintValueInfo;
import com.google.devtools.build.lib.analysis.skylark.SkylarkRuleContext;
import com.google.devtools.build.lib.analysis.test.InstrumentedFilesCollector.InstrumentationSpec;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.skylarkbuildapi.test.CoverageCommonApi;
import com.google.devtools.build.lib.skylarkbuildapi.test.InstrumentedFilesInfoApi;
import com.google.devtools.build.lib.syntax.EvalException;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Sequence;
import com.google.devtools.build.lib.syntax.Starlark;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;

/** Helper functions for Starlark to access coverage-related infrastructure. */
public class CoverageCommon implements CoverageCommonApi<ConstraintValueInfo, SkylarkRuleContext> {

  @Override
  @SuppressWarnings("unchecked") // Casting extensions param is verified by Starlark interpreter.
  public InstrumentedFilesInfoApi instrumentedFilesInfo(
      SkylarkRuleContext skylarkRuleContext,
      Sequence<?> sourceAttributes, // <String>
      Sequence<?> dependencyAttributes, // <String>
      Object extensions)
      throws EvalException {
    List<String> extensionsList =
        extensions == Starlark.NONE
            ? null
            : Sequence.castList((List<?>) extensions, String.class, "extensions");

    return createInstrumentedFilesInfo(
        skylarkRuleContext.getRuleContext(),
        sourceAttributes.getContents(String.class, "source_attributes"),
        dependencyAttributes.getContents(String.class, "dependency_attributes"),
        extensionsList);
  }

  /**
   * Returns a {@link InstrumentedFilesInfo} for the rule defined by the given rule context and
   * various named parameters that define the "instrumentation specification" of the rule. For
   * example, the instrumented sources are determined given the values of the attributes named in
   * {@code sourceAttributes} given by the {@code ruleContext}.
   *
   * @param ruleContext the rule context
   * @param sourceAttributes a list of attribute names which contain source files for the rule
   * @param dependencyAttributes a list of attribute names which contain dependencies that might
   *     propagate instances of {@link InstrumentedFilesInfo}
   * @param extensions file extensions used to filter files from source_attributes. If null, all
   *     files on the source attributes will be treated as instrumented. Otherwise, only files with
   *     extensions listed in {@code extensions} will be used
   */
  public static InstrumentedFilesInfo createInstrumentedFilesInfo(
      RuleContext ruleContext,
      List<String> sourceAttributes,
      List<String> dependencyAttributes,
      @Nullable List<String> extensions) {
    FileTypeSet fileTypeSet = FileTypeSet.ANY_FILE;
    if (extensions != null) {
      if (extensions.isEmpty()) {
        fileTypeSet = FileTypeSet.NO_FILE;
      } else {
        FileType[] fileTypes = new FileType[extensions.size()];
        Arrays.setAll(fileTypes, i -> FileType.of(extensions.get(i)));
        fileTypeSet = FileTypeSet.of(fileTypes);
      }
    }
    InstrumentationSpec instrumentationSpec =
        new InstrumentationSpec(fileTypeSet)
            .withSourceAttributes(sourceAttributes.toArray(new String[0]))
            .withDependencyAttributes(dependencyAttributes.toArray(new String[0]));
    return InstrumentedFilesCollector.collectTransitive(
        ruleContext,
        instrumentationSpec,
        /* reportedToActualSources= */ NestedSetBuilder.create(Order.STABLE_ORDER));
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<coverage_common>");
  }
}
