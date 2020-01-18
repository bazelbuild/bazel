// Copyright 2017 The Bazel Authors. All rights reserved.
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
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.events.Location;
import com.google.devtools.build.lib.util.SpellChecker;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;

/** A syntax-tree-walking evaluator for StarlarkFunction bodies. */
final class Eval {

  private Eval() {} // uninstantiable

  // ---- entry point ----

  // Called from StarlarkFunction.fastcall.
  static Object execFunctionBody(StarlarkThread.CallFrame fr, List<Statement> statements)
      throws EvalException, InterruptedException {
    checkInterrupt();
    execStatements(fr, statements, /*indented=*/ false);
    return fr.result;
  }

  private static StarlarkFunction fn(StarlarkThread.CallFrame fr) {
    return (StarlarkFunction) fr.fn;
  }

  private static TokenKind execStatements(
      StarlarkThread.CallFrame fr, List<Statement> statements, boolean indented)
      throws EvalException, InterruptedException {
    boolean isToplevelFunction = fn(fr).isToplevel;

    // Hot code path, good chance of short lists which don't justify the iterator overhead.
    for (int i = 0; i < statements.size(); i++) {
      Statement stmt = statements.get(i);
      TokenKind flow = exec(fr, stmt);
      if (flow != TokenKind.PASS) {
        return flow;
      }

      // Hack for SkylarkImportLookupFunction's "export" semantics.
      // We enable it only for statements outside any function (isToplevelFunction)
      // and outside any if- or for- statements (!indented).
      if (isToplevelFunction && !indented && fr.thread.postAssignHook != null) {
        if (stmt instanceof AssignmentStatement) {
          AssignmentStatement assign = (AssignmentStatement) stmt;
          for (Identifier id : Identifier.boundIdentifiers(assign.getLHS())) {
            String name = id.getName();
            Object value = fn(fr).getModule().lookup(name);
            fr.thread.postAssignHook.assign(name, value);
          }
        }
      }
    }
    return TokenKind.PASS;
  }

  private static void execAssignment(StarlarkThread.CallFrame fr, AssignmentStatement node)
      throws EvalException, InterruptedException {
    if (node.isAugmented()) {
      execAugmentedAssignment(fr, node);
    } else {
      Object rvalue = eval(fr, node.getRHS());
      try {
        assign(fr, node.getLHS(), rvalue);
      } catch (EvalException ex) {
        // TODO(adonovan): use location of = operator.
        throw ex.ensureLocation(node.getStartLocation());
      }
    }
  }

  private static TokenKind execFor(StarlarkThread.CallFrame fr, ForStatement node)
      throws EvalException, InterruptedException {
    Object o = eval(fr, node.getCollection());
    Iterable<?> seq = Starlark.toIterable(o);
    EvalUtils.lock(o, node.getStartLocation());
    try {
      for (Object it : seq) {
        assign(fr, node.getLHS(), it);

        switch (execStatements(fr, node.getBlock(), /*indented=*/ true)) {
          case PASS:
          case CONTINUE:
            // Stay in loop.
            checkInterrupt();
            continue;
          case BREAK:
            // Finish loop, execute next statement after loop.
            return TokenKind.PASS;
          case RETURN:
            // Finish loop, return from function.
            return TokenKind.RETURN;
          default:
            throw new IllegalStateException("unreachable");
        }
      }
    } catch (EvalException ex) {
      throw ex.ensureLocation(node.getLHS().getStartLocation());
    } finally {
      EvalUtils.unlock(o, node.getStartLocation());
    }
    return TokenKind.PASS;
  }

  private static void execDef(StarlarkThread.CallFrame fr, DefStatement node)
      throws EvalException, InterruptedException {
    FunctionSignature sig = node.getSignature();

    // Evaluate default value expressions of optional parameters.
    // They may be discontinuous:
    // def f(a, b=1, *, c, d=2) has a defaults tuple of (1, 2).
    // TODO(adonovan): record the gaps (e.g. c) with a sentinel
    // to simplify Starlark.matchSignature.
    Tuple<Object> defaults = Tuple.empty();
    int ndefaults = node.getSignature().numOptionals();
    if (ndefaults > 0) {
      Object[] array = new Object[ndefaults];
      for (int i = sig.numMandatoryPositionals(), j = 0; i < sig.numParameters(); i++) {
        Expression expr = node.getParameters().get(i).getDefaultValue();
        if (expr != null) {
          array[j++] = eval(fr, expr);
        }
      }
      defaults = Tuple.wrap(array);
    }

    assignIdentifier(
        fr,
        node.getIdentifier(),
        new StarlarkFunction(
            node.getIdentifier().getName(),
            node.getIdentifier().getStartLocation(),
            sig,
            defaults,
            node.getStatements(),
            fn(fr).getModule()));
  }

  private static TokenKind execIf(StarlarkThread.CallFrame fr, IfStatement node)
      throws EvalException, InterruptedException {
    boolean cond = Starlark.truth(eval(fr, node.getCondition()));
    if (cond) {
      return execStatements(fr, node.getThenBlock(), /*indented=*/ true);
    } else if (node.getElseBlock() != null) {
      return execStatements(fr, node.getElseBlock(), /*indented=*/ true);
    }
    return TokenKind.PASS;
  }

  private static void execLoad(StarlarkThread.CallFrame fr, LoadStatement node)
      throws EvalException {
    for (LoadStatement.Binding binding : node.getBindings()) {
      Identifier orig = binding.getOriginalName();

      // TODO(adonovan): make this a static check.
      if (orig.isPrivate() && !node.mayLoadInternalSymbols()) {
        throw new EvalException(
            orig.getStartLocation(),
            "symbol '" + orig.getName() + "' is private and cannot be imported.");
      }

      // Load module.
      String moduleName = node.getImport().getValue();
      StarlarkThread.Extension module = fr.thread.getExtension(moduleName);
      if (module == null) {
        throw new EvalException(
            node.getImport().getStartLocation(),
            String.format(
                "file '%s' was not correctly loaded. "
                    + "Make sure the 'load' statement appears in the global scope in your file",
                moduleName));
      }

      // Extract symbol.
      Object value = module.getBindings().get(orig.getName());
      if (value == null) {
        throw new EvalException(
            orig.getStartLocation(),
            String.format(
                "file '%s' does not contain symbol '%s'%s",
                moduleName,
                orig.getName(),
                SpellChecker.didYouMean(orig.getName(), module.getBindings().keySet())));
      }

      // Define module-local variable.
      // TODO(adonovan): eventually the default behavior should be that
      // loads bind file-locally. Either way, the resolver should designate
      // the proper scope of binding.getLocalName() and this should become
      // simply assign(binding.getLocalName(), value).
      // Currently, we update the module but not module.exportedBindings;
      // changing it to fr.locals.put breaks a test. TODO(adonovan): find out why.
      try {
        fn(fr).getModule().put(binding.getLocalName().getName(), value);
      } catch (Mutability.MutabilityException ex) {
        throw new AssertionError(ex);
      }
    }
  }

  private static TokenKind execReturn(StarlarkThread.CallFrame fr, ReturnStatement node)
      throws EvalException, InterruptedException {
    Expression ret = node.getReturnExpression();
    if (ret != null) {
      fr.result = eval(fr, ret);
    }
    return TokenKind.RETURN;
  }

  private static TokenKind exec(StarlarkThread.CallFrame fr, Statement st)
      throws EvalException, InterruptedException {
    if (fr.dbg != null) {
      Location loc = st.getStartLocation();
      fr.setLocation(loc);
      fr.dbg.before(fr.thread, loc); // location is now redundant since it's in the thread
    }

    try {
      return execDispatch(fr, st);
    } catch (EvalException ex) {
      throw maybeTransformException(st, ex);
    }
  }

  private static TokenKind execDispatch(StarlarkThread.CallFrame fr, Statement st)
      throws EvalException, InterruptedException {
    switch (st.kind()) {
      case ASSIGNMENT:
        execAssignment(fr, (AssignmentStatement) st);
        return TokenKind.PASS;
      case EXPRESSION:
        eval(fr, ((ExpressionStatement) st).getExpression());
        return TokenKind.PASS;
      case FLOW:
        return ((FlowStatement) st).getKind();
      case FOR:
        return execFor(fr, (ForStatement) st);
      case DEF:
        execDef(fr, (DefStatement) st);
        return TokenKind.PASS;
      case IF:
        return execIf(fr, (IfStatement) st);
      case LOAD:
        execLoad(fr, (LoadStatement) st);
        return TokenKind.PASS;
      case RETURN:
        return execReturn(fr, (ReturnStatement) st);
    }
    throw new IllegalArgumentException("unexpected statement: " + st.kind());
  }

  /**
   * Updates the environment bindings, and possibly mutates objects, so as to assign the given value
   * to the given expression. May throw an EvalException without location.
   */
  private static void assign(StarlarkThread.CallFrame fr, Expression expr, Object value)
      throws EvalException, InterruptedException {
    if (expr instanceof Identifier) {
      assignIdentifier(fr, (Identifier) expr, value);
    } else if (expr instanceof IndexExpression) {
      Object object = eval(fr, ((IndexExpression) expr).getObject());
      Object key = eval(fr, ((IndexExpression) expr).getKey());
      assignItem(object, key, value);
    } else if (expr instanceof ListExpression) {
      ListExpression list = (ListExpression) expr;
      assignList(fr, list.getElements(), value);
    } else {
      // Not possible for validated ASTs.
      throw Starlark.errorf("cannot assign to '%s'", expr);
    }
  }

  private static void assignIdentifier(StarlarkThread.CallFrame fr, Identifier id, Object value)
      throws EvalException {
    ValidationEnvironment.Scope scope = id.getScope();
    // Legacy hack for incomplete identifier resolution.
    // In a <toplevel> function, assignments to unresolved identifiers
    // update the module, except for load statements and comprehensions,
    // which should both be file-local.
    // Load statements don't yet use assignIdentifier,
    // so we need consider only comprehensions.
    // In effect, we do the missing resolution using fr.compcount.
    if (scope == null) {
      scope =
          fn(fr).isToplevel && fr.compcount == 0
              ? ValidationEnvironment.Scope.Module //
              : ValidationEnvironment.Scope.Local;
    }

    String name = id.getName();
    switch (scope) {
      case Local:
        fr.locals.put(name, value);
        break;
      case Module:
        // Updates a module binding and sets its 'exported' flag.
        // (Only load bindings are not exported.
        // But exportedBindings does at run time what should be done in the resolver.)
        Module module = fn(fr).getModule();
        try {
          module.put(name, value);
          module.exportedBindings.add(name);
        } catch (Mutability.MutabilityException ex) {
          throw new IllegalStateException(ex);
        }
        break;
      default:
        throw new IllegalStateException(scope.toString());
    }
  }

  /**
   * Adds or changes an object-key-value relationship for a list or dict.
   *
   * <p>For a list, the key is an in-range index. For a dict, it is a hashable value.
   *
   * @throws EvalException if the object is not a list or dict
   */
  private static void assignItem(Object object, Object key, Object value) throws EvalException {
    if (object instanceof Dict) {
      @SuppressWarnings("unchecked")
      Dict<Object, Object> dict = (Dict<Object, Object>) object;
      dict.put(key, value, /*loc=*/ null);
    } else if (object instanceof StarlarkList) {
      @SuppressWarnings("unchecked")
      StarlarkList<Object> list = (StarlarkList<Object>) object;
      int index = Starlark.toInt(key, "list index");
      index = EvalUtils.getSequenceIndex(index, list.size());
      list.set(index, value, /*loc=*/ null);
    } else {
      throw Starlark.errorf(
          "can only assign an element in a dictionary or a list, not in a '%s'",
          Starlark.type(object));
    }
  }

  /**
   * Recursively assigns an iterable value to a sequence of assignable expressions. May throw an
   * EvalException without location.
   */
  private static void assignList(StarlarkThread.CallFrame fr, List<Expression> lhs, Object x)
      throws EvalException, InterruptedException {
    // TODO(adonovan): lock/unlock rhs during iteration so that
    // assignments fail when the left side aliases the right,
    // which is a tricky case in Python assignment semantics.
    int nrhs = Starlark.len(x);
    if (nrhs < 0) {
      throw Starlark.errorf("type '%s' is not iterable", Starlark.type(x));
    }
    Iterable<?> rhs = Starlark.toIterable(x); // fails if x is a string
    int len = lhs.size();
    if (len == 0) {
      throw Starlark.errorf(
          "lists or tuples on the left-hand side of assignments must have at least one item");
    }
    if (len != nrhs) {
      throw Starlark.errorf(
          "assignment length mismatch: left-hand side has length %d, but right-hand side evaluates"
              + " to value of length %d",
          len, nrhs);
    }
    int i = 0;
    for (Object item : rhs) {
      assign(fr, lhs.get(i), item);
      i++;
    }
  }

  private static void execAugmentedAssignment(StarlarkThread.CallFrame fr, AssignmentStatement stmt)
      throws EvalException, InterruptedException {
    Expression lhs = stmt.getLHS();
    TokenKind op = stmt.getOperator();
    Expression rhs = stmt.getRHS();
    // TODO(adonovan): don't materialize Locations before an error has occurred.
    // (Requires syntax tree to record offsets and defer Location conversion.)
    Location loc = stmt.getStartLocation(); // TODO(adonovan): use operator location

    if (lhs instanceof Identifier) {
      Object x = eval(fr, lhs);
      Object y = eval(fr, rhs);
      Object z = inplaceBinaryOp(fr, op, x, y, loc);
      assignIdentifier(fr, (Identifier) lhs, z);
    } else if (lhs instanceof IndexExpression) {
      // object[index] op= y
      // The object and key should be evaluated only once, so we don't use lhs.eval().
      IndexExpression index = (IndexExpression) lhs;
      Object object = eval(fr, index.getObject());
      Object key = eval(fr, index.getKey());
      Object x = EvalUtils.index(fr.thread.mutability(), fr.thread.getSemantics(), object, key);
      // Evaluate rhs after lhs.
      Object y = eval(fr, rhs);
      Object z = inplaceBinaryOp(fr, op, x, y, loc);
      try {
        assignItem(object, key, z);
      } catch (EvalException ex) {
        throw ex.ensureLocation(loc);
      }
    } else if (lhs instanceof ListExpression) {
      throw new EvalException(loc, "cannot perform augmented assignment on a list literal");
    } else {
      // Not possible for validated ASTs.
      throw new EvalException(loc, "cannot perform augmented assignment on '" + lhs + "'");
    }
  }

  private static Object inplaceBinaryOp(
      StarlarkThread.CallFrame fr, TokenKind op, Object x, Object y, Location location)
      throws EvalException {
    // list += iterable  behaves like  list.extend(iterable)
    // TODO(b/141263526): following Python, allow list+=iterable (but not list+iterable).
    if (op == TokenKind.PLUS && x instanceof StarlarkList && y instanceof StarlarkList) {
      StarlarkList<?> list = (StarlarkList) x;
      list.extend(y);
      return list;
    }
    return EvalUtils.binaryOp(op, x, y, fr.thread, location);
  }

  // ---- expressions ----

  private static Object eval(StarlarkThread.CallFrame fr, Expression expr)
      throws EvalException, InterruptedException {
    // TODO(adonovan): don't push and pop all the time. We should only need the stack of function
    // call frames, and we should recycle them.
    // TODO(adonovan): put the StarlarkThread into the Java thread-local store
    // once only, in push, and undo this in pop.
    try {
      if (Callstack.enabled) {
        Callstack.push(expr);
      }
      try {
        return doEval(fr, expr);
      } catch (EvalException ex) {
        throw maybeTransformException(expr, ex);
      }
    } finally {
      if (Callstack.enabled) {
        Callstack.pop();
      }
    }
  }

  private static Object doEval(StarlarkThread.CallFrame fr, Expression expr)
      throws EvalException, InterruptedException {
    switch (expr.kind()) {
      case BINARY_OPERATOR:
        {
          BinaryOperatorExpression binop = (BinaryOperatorExpression) expr;
          Object x = eval(fr, binop.getX());
          // AND and OR require short-circuit evaluation.
          switch (binop.getOperator()) {
            case AND:
              return Starlark.truth(x) ? eval(fr, binop.getY()) : x;
            case OR:
              return Starlark.truth(x) ? x : eval(fr, binop.getY());
            default:
              Object y = eval(fr, binop.getY());
              // TODO(adonovan): use operator location
              return EvalUtils.binaryOp(
                  binop.getOperator(), x, y, fr.thread, binop.getStartLocation());
          }
        }

      case COMPREHENSION:
        return evalComprehension(fr, (Comprehension) expr);

      case CONDITIONAL:
        {
          ConditionalExpression cond = (ConditionalExpression) expr;
          Object v = eval(fr, cond.getCondition());
          return eval(fr, Starlark.truth(v) ? cond.getThenCase() : cond.getElseCase());
        }

      case DICT_EXPR:
        {
          DictExpression dictexpr = (DictExpression) expr;
          Dict<Object, Object> dict = Dict.of(fr.thread.mutability());
          for (DictExpression.Entry entry : dictexpr.getEntries()) {
            Object k = eval(fr, entry.getKey());
            Object v = eval(fr, entry.getValue());
            int before = dict.size();
            try {
              dict.put(k, v, /*loc=*/ null);
            } catch (EvalException ex) {
              // TODO(adonovan): use colon location
              throw ex.ensureLocation(entry.getKey().getStartLocation());
            }
            if (dict.size() == before) {
              // TODO(adonovan): use colon location
              throw new EvalException(
                  entry.getKey().getStartLocation(),
                  "Duplicated key " + Starlark.repr(k) + " when creating dictionary");
            }
          }
          return dict;
        }

      case DOT:
        {
          DotExpression dot = (DotExpression) expr;
          Object object = eval(fr, dot.getObject());
          String name = dot.getField().getName();
          try {
            Object result = EvalUtils.getAttr(fr.thread, object, name);
            if (result == null) {
              throw EvalUtils.getMissingAttrException(object, name, fr.thread.getSemantics());
            }
            return result;
          } catch (EvalException ex) {
            throw ex.ensureLocation(dot.getStartLocation());
          }
        }

      case CALL:
        {
          checkInterrupt();

          CallExpression call = (CallExpression) expr;
          Object fn = eval(fr, call.getFunction());

          // StarStar and Star args are guaranteed to be last, if they occur.
          ImmutableList<Argument> arguments = call.getArguments();
          int n = arguments.size();
          Argument.StarStar starstar = null;
          if (n > 0 && arguments.get(n - 1) instanceof Argument.StarStar) {
            starstar = (Argument.StarStar) arguments.get(n - 1);
            n--;
          }
          Argument.Star star = null;
          if (n > 0 && arguments.get(n - 1) instanceof Argument.Star) {
            star = (Argument.Star) arguments.get(n - 1);
            n--;
          }
          // Inv: n = |positional| + |named|

          // Allocate assuming no *args/**kwargs.
          int npos = call.getNumPositionalArguments();
          int i;

          // f(expr) -- positional args
          Object[] positional = npos == 0 ? EMPTY : new Object[npos];
          for (i = 0; i < npos; i++) {
            Argument arg = arguments.get(i);
            Object value = eval(fr, arg.getValue());
            positional[i] = value;
          }

          // f(id=expr) -- named args
          Object[] named = n == npos ? EMPTY : new Object[2 * (n - npos)];
          for (int j = 0; i < n; i++) {
            Argument.Keyword arg = (Argument.Keyword) arguments.get(i);
            Object value = eval(fr, arg.getValue());
            named[j++] = arg.getName();
            named[j++] = value;
          }

          // f(*args) -- varargs
          if (star != null) {
            Object value = eval(fr, star.getValue());
            if (!(value instanceof StarlarkIterable)) {
              throw new EvalException(
                  star.getStartLocation(),
                  "argument after * must be an iterable, not " + Starlark.type(value));
            }
            // TODO(adonovan): opt: if value.size is known, preallocate (and skip if empty).
            ArrayList<Object> list = new ArrayList<>();
            Collections.addAll(list, positional);
            Iterables.addAll(list, ((Iterable<?>) value));
            positional = list.toArray();
          }

          // f(**kwargs)
          if (starstar != null) {
            Object value = eval(fr, starstar.getValue());
            if (!(value instanceof Dict)) {
              throw new EvalException(
                  starstar.getStartLocation(),
                  "argument after ** must be a dict, not " + Starlark.type(value));
            }
            Dict<?, ?> kwargs = (Dict<?, ?>) value;
            int j = named.length;
            named = Arrays.copyOf(named, j + 2 * kwargs.size());
            for (Map.Entry<?, ?> e : kwargs.entrySet()) {
              if (!(e.getKey() instanceof String)) {
                throw new EvalException(
                    starstar.getStartLocation(),
                    "keywords must be strings, not " + Starlark.type(e.getKey()));
              }
              named[j++] = e.getKey();
              named[j++] = e.getValue();
            }
          }

          Location loc = call.getStartLocation(); // TODO(adonovan): use call lparen
          fr.setLocation(loc);
          try {
            return Starlark.fastcall(fr.thread, fn, positional, named);
          } catch (EvalException ex) {
            throw ex.ensureLocation(loc);
          }
        }

      case IDENTIFIER:
        {
          Identifier id = (Identifier) expr;
          String name = id.getName();
          if (id.getScope() == null) {
            // Legacy behavior, to be removed.
            Object result = fr.locals.get(name);
            if (result != null) {
              return result;
            }
            result = fn(fr).getModule().get(name);
            if (result != null) {
              return result;
            }
            String error =
                ValidationEnvironment.createInvalidIdentifierException(
                    name, fr.thread.getVariableNames());
            throw new EvalException(id.getStartLocation(), error);
          }

          Object result;
          switch (id.getScope()) {
            case Local:
              result = fr.locals.get(name);
              break;
            case Module:
              result = fn(fr).getModule().lookup(name);
              break;
            case Universe:
              // TODO(laurentlb): look only at universe.
              result = fn(fr).getModule().get(name);
              break;
            default:
              throw new IllegalStateException(id.getScope().toString());
          }
          if (result == null) {
            // Since Scope was set, we know that the variable is defined in the scope.
            // However, the assignment was not yet executed.
            String error = ValidationEnvironment.getErrorForObsoleteThreadLocalVars(id.getName());
            if (error == null) {
              error =
                  id.getScope().getQualifier()
                      + " variable '"
                      + name
                      + "' is referenced before assignment.";
            }
            throw new EvalException(id.getStartLocation(), error);
          }
          return result;
        }

      case INDEX:
        {
          IndexExpression index = (IndexExpression) expr;
          Object object = eval(fr, index.getObject());
          Object key = eval(fr, index.getKey());
          try {
            return EvalUtils.index(fr.thread.mutability(), fr.thread.getSemantics(), object, key);
          } catch (EvalException ex) {
            // TODO(adonovan): use location of lbracket token
            throw ex.ensureLocation(index.getStartLocation());
          }
        }

      case INTEGER_LITERAL:
        return ((IntegerLiteral) expr).getValue();

      case LIST_EXPR:
        {
          ListExpression list = (ListExpression) expr;
          int n = list.getElements().size();
          Object[] array = new Object[n];
          for (int i = 0; i < n; i++) {
            array[i] = eval(fr, list.getElements().get(i));
          }
          return list.isTuple()
              ? Tuple.wrap(array)
              : StarlarkList.wrap(fr.thread.mutability(), array);
        }

      case SLICE:
        {
          SliceExpression slice = (SliceExpression) expr;
          Object x = eval(fr, slice.getObject());
          Object start = slice.getStart() == null ? Starlark.NONE : eval(fr, slice.getStart());
          Object stop = slice.getStop() == null ? Starlark.NONE : eval(fr, slice.getStop());
          Object step = slice.getStep() == null ? Starlark.NONE : eval(fr, slice.getStep());
          try {
            return Starlark.slice(fr.thread.mutability(), x, start, stop, step);
          } catch (EvalException ex) {
            // TODO(adonovan): use lbracket location
            throw ex.ensureLocation(slice.getStartLocation());
          }
        }

      case STRING_LITERAL:
        return ((StringLiteral) expr).getValue();

      case UNARY_OPERATOR:
        {
          UnaryOperatorExpression unop = (UnaryOperatorExpression) expr;
          Object x = eval(fr, unop.getX());
          try {
            return EvalUtils.unaryOp(unop.getOperator(), x);
          } catch (EvalException ex) {
            throw ex.ensureLocation(unop.getStartLocation());
          }
        }
    }
    throw new IllegalArgumentException("unexpected expression: " + expr.kind());
  }

  private static Object evalComprehension(StarlarkThread.CallFrame fr, Comprehension comp)
      throws EvalException, InterruptedException {
    final Dict<Object, Object> dict = comp.isDict() ? Dict.of(fr.thread.mutability()) : null;
    final ArrayList<Object> list = comp.isDict() ? null : new ArrayList<>();

    // Save previous value (if any) of local variables bound in a 'for' clause
    // so we can restore them later.
    // TODO(adonovan): throw all this away when we implement flat environments.
    List<Object> saved = new ArrayList<>(); // alternating keys and values
    for (Comprehension.Clause clause : comp.getClauses()) {
      if (clause instanceof Comprehension.For) {
        for (Identifier ident :
            Identifier.boundIdentifiers(((Comprehension.For) clause).getVars())) {
          String name = ident.getName();
          Object value = fr.locals.get(ident.getName()); // may be null
          saved.add(name);
          saved.add(value);
        }
      }
    }
    fr.compcount++;

    // The Lambda class serves as a recursive lambda closure.
    class Lambda {
      // execClauses(index) recursively executes the clauses starting at index,
      // and finally evaluates the body and adds its value to the result.
      void execClauses(int index) throws EvalException, InterruptedException {
        checkInterrupt();

        // recursive case: one or more clauses
        if (index < comp.getClauses().size()) {
          Comprehension.Clause clause = comp.getClauses().get(index);
          if (clause instanceof Comprehension.For) {
            Comprehension.For forClause = (Comprehension.For) clause;

            Object iterable = eval(fr, forClause.getIterable());
            Location loc = comp.getStartLocation(); // TODO(adonovan): use location of 'for' token
            Iterable<?> listValue = Starlark.toIterable(iterable);
            // TODO(adonovan): lock should not need loc.
            EvalUtils.lock(iterable, loc);
            try {
              for (Object elem : listValue) {
                assign(fr, forClause.getVars(), elem);
                execClauses(index + 1);
              }
            } catch (EvalException ex) {
              throw ex.ensureLocation(loc);
            } finally {
              EvalUtils.unlock(iterable, loc);
            }

          } else {
            Comprehension.If ifClause = (Comprehension.If) clause;
            if (Starlark.truth(eval(fr, ifClause.getCondition()))) {
              execClauses(index + 1);
            }
          }
          return;
        }

        // base case: evaluate body and add to result.
        if (dict != null) {
          DictExpression.Entry body = (DictExpression.Entry) comp.getBody();
          Object k = eval(fr, body.getKey());
          EvalUtils.checkHashable(k);
          Object v = eval(fr, body.getValue());
          try {
            dict.put(k, v, /*loc=*/ null);
          } catch (EvalException ex) {
            // TODO(adonovan): use colon location
            throw ex.ensureLocation(comp.getStartLocation());
          }
        } else {
          list.add(eval(fr, ((Expression) comp.getBody())));
        }
      }
    }
    new Lambda().execClauses(0);
    fr.compcount--;

    // Restore outer scope variables.
    // This loop implicitly undefines comprehension variables.
    for (int i = 0; i != saved.size(); ) {
      String name = (String) saved.get(i++);
      Object value = saved.get(i++);
      if (value != null) {
        fr.locals.put(name, value);
      } else {
        fr.locals.remove(name);
      }
    }

    return comp.isDict() ? dict : StarlarkList.copyOf(fr.thread.mutability(), list);
  }

  private static final Object[] EMPTY = {};

  /** Returns an exception which should be thrown instead of the original one. */
  private static EvalException maybeTransformException(Node node, EvalException original) {
    // TODO(adonovan): the only place that should be doing this is Starlark.fastcall,
    // and it should grab the entire callstack from the thread at that moment.

    // If there is already a non-empty stack trace, we only add this node iff it describes a
    // new scope (e.g. CallExpression).
    if (original instanceof EvalExceptionWithStackTrace) {
      EvalExceptionWithStackTrace real = (EvalExceptionWithStackTrace) original;
      if (node instanceof CallExpression) {
        real.registerNode(node);
      }
      return real;
    }

    if (original.canBeAddedToStackTrace()) {
      return new EvalExceptionWithStackTrace(original, node);
    } else {
      return original;
    }
  }

  private static void checkInterrupt() throws InterruptedException {
    if (Thread.interrupted()) {
      throw new InterruptedException();
    }
  }
}
