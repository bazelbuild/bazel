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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.events.Location;
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

  // A function signature, including defaults and types
  // never null after it is configured
  protected FunctionSignature.WithValues<Object, SkylarkType> signature;

  // Number of regular named parameters (excluding *p and **p) in the
  // equivalent Python function definition).
  private final List<String> parameters;

  // Number of leading "parameters" which are mandatory
  private final int numMandatoryParameters;

  // True if this function requires all arguments to be named
  // TODO(bazel-team): replace this by a count of arguments before the * with optional arg,
  // in the style Python 3 or PEP 3102.
  private final boolean onlyNamedArguments;

  // Location of the function definition, or null for builtin functions.
  protected final Location location;

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
    this(name, parameters, numMandatoryParameters, onlyNamedArguments, null);
  }

  protected MixedModeFunction(String name,
                              Iterable<String> parameters,
                              int numMandatoryParameters,
                              boolean onlyNamedArguments,
                              Location location) {
    super(name);
    this.parameters = ImmutableList.copyOf(parameters);
    this.numMandatoryParameters = numMandatoryParameters;
    this.onlyNamedArguments = onlyNamedArguments;
    this.location = location;

    // Fake a signature from the above
    this.signature = FunctionSignature.WithValues.<Object, SkylarkType>create(
        FunctionSignature.of(numMandatoryParameters, this.parameters.toArray(new String[0])));
  }


  /** Create a function using a signature with defaults */
  public MixedModeFunction(String name,
      FunctionSignature.WithValues<Object, SkylarkType> signature,
      Location location) {
    super(name);

    // TODO(bazel-team): lift the following limitations, by actually implementing
    // the full function call protocol.
    FunctionSignature sig = signature.getSignature();
    FunctionSignature.Shape shape = sig.getShape();
    Preconditions.checkArgument(!shape.hasKwArg() && !shape.hasStarArg()
        && shape.getNamedOnly() == 0, "no star, star-star or named-only parameters (for now)");

    this.signature = signature;
    this.parameters = ImmutableList.copyOf(sig.getNames());
    this.numMandatoryParameters = shape.getMandatoryPositionals();
    this.onlyNamedArguments = false;
    this.location = location;
  }

  /** Create a function using a signature without defaults */
  public MixedModeFunction(String name, FunctionSignature signature) {
    this(name, FunctionSignature.WithValues.<Object, SkylarkType>create(signature), null);
  }

  @Override
  public Object call(List<Object> args,
                     Map<String, Object> kwargs,
                     FuncallExpression ast,
                     Environment env)
      throws EvalException, InterruptedException {

    // ast is null when called from Java (as there's no Skylark call site).
    Location loc = ast == null ? location : ast.getLocation();
    if (onlyNamedArguments && args.size() > 0) {
      throw new EvalException(loc,
          getSignature() + " does not accept positional arguments");
    }

    if (kwargs == null) {
      kwargs = ImmutableMap.<String, Object>of();
    }

    int numParams = parameters.size();
    int numArgs = args.size();
    Object[] namedArguments = new Object[numParams];

    // first, positional arguments:
    if (numArgs > numParams) {
      throw new EvalException(loc,
          "too many positional arguments in call to " + getSignature());
    }
    for (int ii = 0; ii < numArgs; ++ii) {
      namedArguments[ii] = args.get(ii);
    }

    // TODO(bazel-team): here, support *varargs splicing

    // second, keyword arguments:
    for (Map.Entry<String, Object> entry : kwargs.entrySet()) {
      String keyword = entry.getKey();
      int pos = parameters.indexOf(keyword);
      if (pos == -1) {
        throw new EvalException(loc,
            "unexpected keyword '" + keyword
            + "' in call to " + getSignature());
      } else {
        if (namedArguments[pos] != null) {
          throw new EvalException(loc, getSignature()
              + " got multiple values for keyword argument '" + keyword + "'");
        }
        namedArguments[pos] = kwargs.get(keyword);
      }
    }

    // third, check mandatory parameters:
    for (int ii = 0; ii < numMandatoryParameters; ++ii) {
      if (namedArguments[ii] == null) {
        throw new EvalException(loc,
            getSignature() + " received insufficient arguments");
      }
    }

    // fourth, fill in defaults from the signature, if any
    List<Object> defaults = signature.getDefaultValues();
    if (defaults != null) {
      int jj = 0;
      for (int ii = numMandatoryParameters; ii < numParams; ++ii) {
        if (namedArguments[ii] == null) {
          namedArguments[ii] = defaults.get(jj);
        }
        jj++;
      }
    }

    try {
      return call(namedArguments, ast, env);
    } catch (ConversionException | IllegalArgumentException | IllegalStateException
        | ClassCastException e) {
      throw new EvalException(loc, e.getMessage());
    }
  }

  /**
   * Like Function.call, but generalised to support Python-style mixed-mode
   * keyword and positional parameter passing.
   *
   * @param args an array of argument values corresponding to the list
   *        of named parameters passed to the constructor.
   */
  protected Object call(Object[] args, FuncallExpression ast)
      throws EvalException, ConversionException, InterruptedException {
    throw new UnsupportedOperationException("Method not overridden");
  }

  /**
   * Override this method instead of the one above, if you need to access
   * the environment.
   */
  protected Object call(Object[] args, FuncallExpression ast, Environment env)
      throws EvalException, ConversionException, InterruptedException {
    return call(args, ast);
  }

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
