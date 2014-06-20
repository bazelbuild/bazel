// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.syntax;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.packages.Type.ConversionException;

import java.util.List;
import java.util.Map;

/**
 * Abstract implementation of Function for functions that accept a mixture of
 * positional and keyword parameters, as in Python.
 */
public abstract class MixedModeFunction extends AbstractFunction {

  // Nomenclature:
  // "Parameters" are formal parameters of a function definition.
  // "Arguments" are actual parameters supplied at the call site.

  // Number of regular named parameters (excluding *p and **p) in the
  // equivalent Python function definition).
  private final List<String> parameters;

  // Number of leading "parameters" which are mandatory
  private final int numMandatoryParameters;

  // True if this function requires all arguments to be named
  private final boolean onlyNamedArguments;


  /**
   * Constructs an instance of Function that supports Python-style mixed-mode
   * parameter passing.
   *
   * @param parameters a list of named parameters
   * @param numMandatoryParameters the number of leading parameters which are
   *        considered mandatory; the remaining ones may be omitted, in which
   *        case they will have the default value of null.
   */
  public MixedModeFunction(String name,
                           Iterable<String> parameters,
                           int numMandatoryParameters,
                           boolean onlyNamedArguments) {
    super(name);
    this.parameters = ImmutableList.copyOf(parameters);
    this.numMandatoryParameters = numMandatoryParameters;
    this.onlyNamedArguments = onlyNamedArguments;
  }

  @Override
  public Object call(List<Object> args,
                     Map<String, Object> kwargs,
                     FuncallExpression ast,
                     Environment env)
      throws EvalException, InterruptedException {

    if (onlyNamedArguments && args.size() > 0) {
      throw new EvalException(ast.getLocation(),
          getSignature() + " does not accept positional arguments");
    }

    int numParams = parameters.size();
    int numArgs = args.size();
    Object[] namedArguments = new Object[numParams];

    // first, positional arguments:
    if (numArgs > numParams) {
      throw new EvalException(ast.getLocation(),
          "too many arguments in call to " + getSignature());
    }
    for (int ii = 0; ii < numArgs && ii < numParams; ++ii) {
      namedArguments[ii] = args.get(ii);
    }

    // second, keyword arguments:
    for (Map.Entry<String, Object> entry : kwargs.entrySet()) {
      String keyword = entry.getKey();
      int pos = parameters.indexOf(keyword);
      if (pos == -1) {
        throw new EvalException(ast.getLocation(),
                                "unexpected keyword '" + keyword
                                + "' in call to " + getSignature());
      } else {
        if (namedArguments[pos] != null) {
          throw new EvalException(ast.getLocation(), getSignature()
              + " got multiple values for keyword argument '" + keyword + "'");
        }
        namedArguments[pos] = kwargs.get(keyword);
      }
    }

    // third, defaults:
    for (int ii = 0; ii < numMandatoryParameters; ++ii) {
      if (namedArguments[ii] == null) {
        throw new EvalException(ast.getLocation(),
            getSignature() + " received insufficient arguments");
      }
    }
    // (defaults are always null so nothing extra to do here.)

    try {
      return call(namedArguments,
                  null,
                  null,
                  ast);
    } catch (ConversionException | IllegalArgumentException | IllegalStateException
        | ClassCastException e) {
      throw new EvalException(ast.getLocation(), e.getMessage());
    }
  }

  // TODO(bazel-team): Clean up positionalArguments and keywordArguments.
  /**
   * Like Function.call, but generalised to support Python-style mixed-mode
   * keyword and positional parameter passing.
   *
   * @param namedArguments an array of argument values corresponding to the list
   *        of named parameters passed to the constructor.
   * @param positionalArguments a list of surplus positional arguments
   *        (if this function supports it, otherwise null).
   * @param keywordArguments a dictionary of surplus keyword arguments
   *        (if this function supports it; otherwise null)
   */
  public abstract Object call(Object[] namedArguments,
                              List<Object> positionalArguments,
                              Map<String, Object> keywordArguments,
                              FuncallExpression ast)
      throws EvalException, ConversionException, InterruptedException;

  /**
   * Render this object in the form of an equivalent Python function signature.
   */
  public String getSignature() {
    StringBuffer sb = new StringBuffer();
    sb.append(getName()).append('(');
    int ii = 0;
    int len = parameters.size();
    for (; ii < len; ++ii) {
      String parameter = parameters.get(ii);
      if (ii > 0) {
        sb.append(", ");
      }
      sb.append(parameter);
      if (ii >= numMandatoryParameters) {
        sb.append(" = null");
      }
    }
    sb.append(')');
    return sb.toString();
  }

}
