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
import com.google.devtools.build.lib.cmdline.BazelModuleContext;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.FunctionDeprecationInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.FunctionParamInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.FunctionReturnInfo;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.OriginKey;
import com.google.devtools.build.skydoc.rendering.proto.StardocOutputProtos.StarlarkFunctionInfo;
import com.google.devtools.starlark.common.DocstringUtils;
import com.google.devtools.starlark.common.DocstringUtils.DocstringInfo;
import com.google.devtools.starlark.common.DocstringUtils.DocstringParseError;
import com.google.devtools.starlark.common.DocstringUtils.ParameterDoc;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import javax.annotation.Nullable;
import net.starlark.java.eval.StarlarkFunction;

/** Contains a number of utility methods for functions and parameters. */
public final class StarlarkFunctionInfoExtractor {
  private final LabelRenderer labelRenderer;

  private StarlarkFunctionInfoExtractor(LabelRenderer labelRenderer) {
    this.labelRenderer = labelRenderer;
  }

  /**
   * Create and return a {@link StarlarkFunctionInfo} object encapsulating information obtained from
   * the given function and from its parsed docstring.
   *
   * @param functionName the name of the function in the target scope. (Note this is not necessarily
   *     the original exported function name; the function may have been renamed in the target
   *     Starlark file's scope)
   * @param fn the function object
   * @param withOriginKey set the {@link OriginKey} for the original name and original module where
   *     the function was defined
   * @param labelRenderer a string renderer for {@link Label} values in argument defaults and for
   *     the {@link OriginKey}'s file
   * @throws com.google.devtools.build.skydoc.rendering.DocstringParseException if the function's
   *     docstring is malformed
   */
  // TODO(arostovtsev): remove withOriginKey parameter and always export the origin key after we
  // remove the legacy Stardoc extractor.
  public static StarlarkFunctionInfo fromNameAndFunction(
      String functionName, StarlarkFunction fn, boolean withOriginKey, LabelRenderer labelRenderer)
      throws DocstringParseException {
    return new StarlarkFunctionInfoExtractor(labelRenderer)
        .extract(functionName, fn, withOriginKey);
  }

  private StarlarkFunctionInfo extract(
      String functionName, StarlarkFunction fn, boolean withOriginKey)
      throws DocstringParseException {
    Map<String, String> paramNameToDocMap = Maps.newLinkedHashMap();
    StarlarkFunctionInfo.Builder functionInfoBuilder =
        StarlarkFunctionInfo.newBuilder().setFunctionName(functionName);
    if (withOriginKey) {
      functionInfoBuilder.setOriginKey(getFunctionOriginKey(fn));
    }

    String doc = fn.getDocumentation();

    if (doc != null) {
      List<DocstringParseError> parseErrors = Lists.newArrayList();
      DocstringInfo docstringInfo = DocstringUtils.parseDocstring(doc, parseErrors);
      if (!parseErrors.isEmpty()) {
        throw new DocstringParseException(functionName, fn.getLocation(), parseErrors);
      }
      StringBuilder functionDescription = new StringBuilder(docstringInfo.getSummary());
      if (!docstringInfo.getSummary().isEmpty() && !docstringInfo.getLongDescription().isEmpty()) {
        functionDescription.append("\n\n");
      }
      functionDescription.append(docstringInfo.getLongDescription());
      functionInfoBuilder.setDocString(functionDescription.toString());
      for (ParameterDoc paramDoc : docstringInfo.getParameters()) {
        paramNameToDocMap.put(paramDoc.getParameterName(), paramDoc.getDescription());
      }
      String returns = docstringInfo.getReturns();
      if (!returns.isEmpty()) {
        functionInfoBuilder.setReturn(
            FunctionReturnInfo.newBuilder().setDocString(returns).build());
      }
      String deprecated = docstringInfo.getDeprecated();
      if (!deprecated.isEmpty()) {
        functionInfoBuilder.setDeprecated(
            FunctionDeprecationInfo.newBuilder().setDocString(deprecated).build());
      }
    }
    functionInfoBuilder.addAllParameter(parameterInfos(fn, paramNameToDocMap));
    return functionInfoBuilder.build();
  }

  private FunctionParamInfo forParam(
      String name, Optional<String> docString, @Nullable Object defaultValue) {
    FunctionParamInfo.Builder paramBuilder = FunctionParamInfo.newBuilder().setName(name);
    docString.ifPresent(paramBuilder::setDocString);
    if (defaultValue == null) {
      paramBuilder.setMandatory(true);
    } else {
      paramBuilder.setDefaultValue(labelRenderer.repr(defaultValue)).setMandatory(false);
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

  private ImmutableList<FunctionParamInfo> parameterInfos(
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
        Optional<String> doc = Optional.ofNullable(parameterDoc.get(name));
        info = forParam(name, doc, fn.getDefaultValue(i));
      }
      infos.add(info);
    }
    return infos.build();
  }

  private OriginKey getFunctionOriginKey(StarlarkFunction fn) {
    OriginKey.Builder builder = OriginKey.newBuilder();
    // We can't just `builder.setName(fn.getName())` - fn could be a nested function or a lambda, so
    // fn.getName() may not be a unique name in fn's module. Instead, we look for fn in the module's
    // globals, and if we fail to find it, we leave OriginKey.name unset.
    // For nested functions and lambdas, we could theoretically derive OriginKey.name from
    // fn.getName() and fn.getLocation(), e.g. "<foo at 123:4>". It's unclear how useful this would
    // be in practice; and the location would be highly likely (as compared to docstring content) to
    // change with any edits to the .bzl file, resulting in lots of churn in golden tests.
    for (Map.Entry<String, Object> entry : fn.getModule().getGlobals().entrySet()) {
      if (fn.equals(entry.getValue())) {
        builder.setName(entry.getKey());
        break;
      }
    }
    // TODO(arostovtsev): also recurse into global structs/dicts/lists

    BazelModuleContext moduleContext = BazelModuleContext.of(fn.getModule());
    if (moduleContext != null) {
      builder.setFile(labelRenderer.render(moduleContext.label()));
    }
    return builder.build();
  }
}
