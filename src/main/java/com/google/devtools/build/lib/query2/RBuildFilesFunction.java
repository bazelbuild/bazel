// Copyright 2015 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2;

import com.google.common.base.Function;
import com.google.common.collect.Collections2;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.query2.engine.Callback;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Argument;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.ArgumentType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.VariableContext;
import com.google.devtools.build.lib.vfs.PathFragment;

import java.util.List;

/**
 * An "rbuildfiles" query expression, which computes the set of packages (as represented by their
 * BUILD source file targets) that depend on the given set of files, either as BUILD files directly
 * or as subincludes. Is morally the inverse of the "buildfiles" operator, although that operator
 * takes targets and returns subinclude targets, while this takes files and returns BUILD file
 * targets.
 *
 * <pre>expr ::= RBUILDFILES '(' WORD, ... ')'</pre>
 *
 * <p>This expression can only be used with SkyQueryEnvironment.
 */
public class RBuildFilesFunction implements QueryFunction {

  @Override
  public String getName() {
    return "rbuildfiles";
  }

  @Override
  public int getMandatoryArguments() {
    return 1;
  }

  @Override
  public Iterable<ArgumentType> getArgumentTypes() {
    return Iterables.cycle(ArgumentType.WORD);
  }

  private static final Function<Argument, PathFragment> ARGUMENT_TO_PATH_FRAGMENT =
      new Function<Argument, PathFragment>() {
        @Override
        public PathFragment apply(Argument argument) {
          return new PathFragment(argument.getWord());
        }
      };

  @Override
  @SuppressWarnings("unchecked") // Cast from <Target> to <T>. This will only be used with <Target>.
  public <T> void eval(
      QueryEnvironment<T> env,
      VariableContext<T> context,
      QueryExpression expression,
      List<Argument> args,
      Callback<T> callback) throws QueryException, InterruptedException {
    if (!(env instanceof SkyQueryEnvironment)) {
      throw new QueryException("rbuildfiles can only be used with SkyQueryEnvironment");
    }
    ((SkyQueryEnvironment) env)
        .getRBuildFiles(
            Collections2.transform(args, ARGUMENT_TO_PATH_FRAGMENT), (Callback<Target>) callback);
  }
}
