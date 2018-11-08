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

package com.google.devtools.build.skydoc.rendering;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.syntax.FunctionSignature;
import com.google.devtools.build.lib.syntax.SkylarkType;
import com.google.devtools.build.lib.syntax.StringLiteral;
import com.google.devtools.build.lib.syntax.UserDefinedFunction;
import com.google.devtools.skylark.skylint.DocstringUtils;
import com.google.devtools.skylark.skylint.DocstringUtils.DocstringInfo;
import com.google.devtools.skylark.skylint.DocstringUtils.DocstringParseError;
import com.google.devtools.skylark.skylint.DocstringUtils.ParameterDoc;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/** Encapsulates information about a user-defined Starlark function. */
public class UserDefinedFunctionInfo {

  private final String functionName;
  private final Collection<FunctionParamInfo> parameters;
  private final String docString;

  /**
   * An exception that may be thrown during construction of {@link UserDefinedFunctionInfo} if the
   * function's docstring is malformed.
   */
  public static class DocstringParseException extends Exception {
    public DocstringParseException(
        String functionName, Location definedLocation, List<DocstringParseError> parseErrors) {
      super(getMessage(functionName, definedLocation, parseErrors));
    }

    private static String getMessage(
        String functionName, Location definedLocation, List<DocstringParseError> parseErrors) {
      StringBuilder message = new StringBuilder();
      message.append(
          String.format(
              "Unable to generate documentation for function %s (defined at %s) "
                  + "due to malformed docstring. Parse errors:\n",
              functionName, definedLocation));
      for (DocstringParseError parseError : parseErrors) {
        message.append(
            String.format(
                "  %s line %s: %s\n",
                definedLocation,
                parseError.getLineNumber(),
                parseError.getMessage().replace('\n', ' ')));
      }
      return message.toString();
    }
  }

  /**
   * Create and return a {@link UserDefinedFunctionInfo} object encapsulating information obtained
   * from the given function and from its parsed docstring.
   *
   * @param functionName the name of the function in the target scope. (Note this is not necessarily
   *     the original exported function name; the function may have been renamed in the target
   *     Starlark file's scope)
   * @param userDefinedFunction the raw function object
   * @throws DocstringParseException if the function's docstring is malformed
   */
  public static UserDefinedFunctionInfo fromNameAndFunction(
      String functionName, UserDefinedFunction userDefinedFunction) throws DocstringParseException {
    String functionDescription = "";
    Map<String, String> paramNameToDocMap = Maps.newLinkedHashMap();

    StringLiteral docStringLiteral =
        DocstringUtils.extractDocstring(userDefinedFunction.getStatements());

    if (docStringLiteral != null) {
      List<DocstringParseError> parseErrors = Lists.newArrayList();
      DocstringInfo docstringInfo = DocstringUtils.parseDocstring(docStringLiteral, parseErrors);
      if (!parseErrors.isEmpty()) {
        throw new DocstringParseException(
            functionName, userDefinedFunction.getLocation(), parseErrors);
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

    return new UserDefinedFunctionInfo(
        functionName,
        parameterInfos(userDefinedFunction, paramNameToDocMap),
        functionDescription);
  }

  private static List<FunctionParamInfo> parameterInfos(
      UserDefinedFunction userDefinedFunction,
      Map<String, String> paramNameToDocMap)  {
    FunctionSignature.WithValues<Object, SkylarkType> signature =
        userDefinedFunction.getSignature();
    ImmutableList.Builder<FunctionParamInfo> parameterInfos = ImmutableList.builder();

    List<String> paramNames = signature.getSignature().getNames();
    // Mandatory parameters must always come before optional parameters, so this counts
    // down until all mandatory parameters have been exhausted, and then starts filling in
    // the default parameters accordingly.
    int numMandatoryParamsLeft =
        signature.getDefaultValues() != null
            ? paramNames.size() - signature.getDefaultValues().size()
            : paramNames.size();
    int optionalParamIndex = 0;

    for (String paramName : paramNames) {
      Object defaultParamValue = null;
      String paramDoc = "";
      if (numMandatoryParamsLeft == 0) {
        defaultParamValue = signature.getDefaultValues().get(optionalParamIndex);
        optionalParamIndex++;
      } else {
        numMandatoryParamsLeft--;
      }
      if (paramNameToDocMap.containsKey(paramName)) {
        paramDoc = paramNameToDocMap.get(paramName);
      }
      parameterInfos.add(new FunctionParamInfo(paramName, paramDoc, defaultParamValue));
    }
    return parameterInfos.build();
  }

  private UserDefinedFunctionInfo(
      String functionName, Collection<FunctionParamInfo> parameters, String docString) {
    this.functionName = functionName;
    this.parameters = parameters;
    this.docString = docString;
  }

  /** Returns the raw name of this function, for example, "my_function". */
  public String getName() {
    return functionName;
  }

  /**
   * Returns a collection of {@link FunctionParamInfo} objects, where each encapsulates information
   * about what of the function parameters. Ordering matches the actual Starlark function signature.
   */
  public Collection<FunctionParamInfo> getParameters() {
    return parameters;
  }

  /** Returns the summary form string this function, for example, "my_function(foo, bar)". */
  // TODO(cparsons): Compute summary form in the markdown template, as there should be links
  // between the summary's parameter names and their corresponding documentation.
  @SuppressWarnings("unused") // Used by markdown template.
  public String getSummaryForm() {
    List<String> paramNames =
        parameters.stream().map(param -> param.getName()).collect(Collectors.toList());
    return functionName + "(" + Joiner.on(", ").join(paramNames) + ")";
  }

  /**
   * Returns the portion of the docstring that is not part of any special sections, such as "Args:"
   * or "Returns:". Returns the empty string if there is no docstring literal for this function.
   */
  public String getDocString() {
    return docString;
  }
}
