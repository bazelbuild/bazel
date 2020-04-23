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

package com.google.devtools.build.skydoc.rendering;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.syntax.Printer;
import com.google.devtools.build.lib.syntax.Printer.BasePrinter;
import com.google.devtools.build.lib.syntax.StarlarkFunction;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.FunctionParamInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.StarlarkFunctionInfo;
import com.google.devtools.skylark.common.DocstringUtils;
import com.google.devtools.skylark.common.DocstringUtils.DocstringInfo;
import com.google.devtools.skylark.common.DocstringUtils.DocstringParseError;
import com.google.devtools.skylark.common.DocstringUtils.ParameterDoc;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** Contains a number of utility methods for functions and parameters. */
public final class FunctionUtil {
  /**
   * Create and return a {@link StarlarkFunctionInfo} object encapsulating information obtained from
   * the given function and from its parsed docstring.
   *
   * @param functionName the name of the function in the target scope. (Note this is not necessarily
   *     the original exported function name; the function may have been renamed in the target
   *     Starlark file's scope)
   * @param fn the function object
   * @throws com.google.devtools.build.skydoc.rendering.DocstringParseException if the function's
   *     docstring is malformed
   */
  public static StarlarkFunctionInfo fromNameAndFunction(String functionName, StarlarkFunction fn)
      throws DocstringParseException {
    String functionDescription = "";
    Map<String, String> paramNameToDocMap = Maps.newLinkedHashMap();

    String doc = fn.getDocumentation();
    if (doc != null) {
      List<DocstringParseError> parseErrors = Lists.newArrayList();
      DocstringInfo docstringInfo = DocstringUtils.parseDocstring(doc, parseErrors);
      if (!parseErrors.isEmpty()) {
        throw new DocstringParseException(functionName, fn.getLocation(), parseErrors);
      }
      functionDescription += docstringInfo.getSummary();
      if (!docstringInfo.getSummary().isEmpty() && !docstringInfo.getLongDescription().isEmpty()) {
        functionDescription += "\n\n";
      }
      functionDescription += docstringInfo.getLongDescription();
      for (ParameterDoc paramDoc : docstringInfo.getParameters()) {
        paramNameToDocMap.put(paramDoc.getParameterName(), paramDoc.getDescription());
      }
    }
    List<FunctionParamInfo> paramsInfo = parameterInfos(fn, paramNameToDocMap);
    return StarlarkFunctionInfo.newBuilder()
        .setFunctionName(functionName)
        .setDocString(functionDescription)
        .addAllParameter(paramsInfo)
        .build();
  }

  /** Constructor to be used for normal parameters. */
  public static FunctionParamInfo forParam(
      String name, String docString, @Nullable Object defaultValue) {
    FunctionParamInfo.Builder paramBuilder =
        FunctionParamInfo.newBuilder().setName(name).setDocString(docString);
    if (defaultValue == null) {
      paramBuilder.setMandatory(true);
    } else {
      BasePrinter printer = Printer.getSimplifiedPrinter();
      printer.repr(defaultValue);
      String defaultValueString = printer.toString();

      if (defaultValueString.isEmpty()) {
        defaultValueString = "{unknown object}";
      }
      paramBuilder.setDefaultValue(defaultValueString).setMandatory(false);
    }
    return paramBuilder.build();
  }

  /** Constructor to be used for *args or **kwargs. */
  public static FunctionParamInfo forSpecialParam(String name, String docString) {
    return FunctionParamInfo.newBuilder()
        .setName(name)
        .setDocString(docString)
        .setMandatory(false)
        .build();
  }

  private static List<FunctionParamInfo> parameterInfos(
      StarlarkFunction fn, Map<String, String> parameterDoc) {
    List<String> names = fn.getParameterNames();
    int nparams = names.size();
    int kwargsIndex = fn.hasKwargs() ? --nparams : -1;
    int varargsIndex = fn.hasVarargs() ? --nparams : -1;
    // Inv: nparams is number of regular parameters.

    ImmutableList.Builder<FunctionParamInfo> infos = ImmutableList.builder();
    for (int i = 0; i < names.size(); i++) {
      String name = names.get(i);
      FunctionParamInfo info;
      if (i == varargsIndex) {
        // *args
        String doc = parameterDoc.getOrDefault("*" + name, "");
        info = forSpecialParam(name, doc);
      } else if (i == kwargsIndex) {
        // **kwargs
        String doc = parameterDoc.getOrDefault("**" + name, "");
        info = forSpecialParam(name, doc);
      } else {
        // regular parameter
        String doc = parameterDoc.getOrDefault(name, "");
        info = forParam(name, doc, fn.getDefaultValue(i));
      }
      infos.add(info);
    }
    return infos.build();
  }
}
