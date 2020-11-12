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

package net.starlark.java.eval;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.math.BigInteger;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import net.starlark.java.spelling.SpellChecker;
import net.starlark.java.syntax.Argument;
import net.starlark.java.syntax.AssignmentStatement;
import net.starlark.java.syntax.BinaryOperatorExpression;
import net.starlark.java.syntax.CallExpression;
import net.starlark.java.syntax.Comprehension;
import net.starlark.java.syntax.ConditionalExpression;
import net.starlark.java.syntax.DefStatement;
import net.starlark.java.syntax.DictExpression;
import net.starlark.java.syntax.DotExpression;
import net.starlark.java.syntax.Expression;
import net.starlark.java.syntax.ExpressionStatement;
import net.starlark.java.syntax.FloatLiteral;
import net.starlark.java.syntax.FlowStatement;
import net.starlark.java.syntax.ForStatement;
import net.starlark.java.syntax.Identifier;
import net.starlark.java.syntax.IfStatement;
import net.starlark.java.syntax.IndexExpression;
import net.starlark.java.syntax.IntLiteral;
import net.starlark.java.syntax.ListExpression;
import net.starlark.java.syntax.LoadStatement;
import net.starlark.java.syntax.Location;
import net.starlark.java.syntax.Resolver;
import net.starlark.java.syntax.ReturnStatement;
import net.starlark.java.syntax.SliceExpression;
import net.starlark.java.syntax.Statement;
import net.starlark.java.syntax.StringLiteral;
import net.starlark.java.syntax.TokenKind;
import net.starlark.java.syntax.UnaryOperatorExpression;

final class Eval {

  private Eval() {} // uninstantiable

  // ---- entry point ----

  // Called from StarlarkFunction.fastcall.
  static Object execFunctionBody(StarlarkThread.Frame fr, List<Statement> statements)
      throws EvalException, InterruptedException {
    fr.thread.checkInterrupt();
    execStatements(fr, statements, /*indented=*/ false);
    return fr.result;
  }

  private static StarlarkFunction fn(StarlarkThread.Frame fr) {
    return (StarlarkFunction) fr.fn;
  }

  private static TokenKind execStatements(
      StarlarkThread.Frame fr, List<Statement> statements, boolean indented)
      throws EvalException, InterruptedException {
    boolean isToplevelFunction = fn(fr).isToplevel();

    // Hot code path, good chance of short lists which don't justify the iterator overhead.
    for (int i = 0; i < statements.size(); i++) {
      Statement stmt = statements.get(i);
      TokenKind flow = exec(fr, stmt);
      if (flow != TokenKind.PASS) {
        return flow;
      }

      // Hack for BzlLoadFunction's "export" semantics.
      // We enable it only for statements outside any function (isToplevelFunction)
      // and outside any if- or for- statements (!indented).
      if (isToplevelFunction && !indented && fr.thread.postAssignHook != null) {
        if (stmt instanceof AssignmentStatement) {
          AssignmentStatement assign = (AssignmentStatement) stmt;
          for (Identifier id : Identifier.boundIdentifiers(assign.getLHS())) {
            String name = id.getName();
            Object value = fn(fr).getModule().getGlobal(name);
            fr.thread.postAssignHook.assign(name, value);
          }
        }
      }
    }
    return TokenKind.PASS;
  }

  private static void execAssignment(StarlarkThread.Frame fr, AssignmentStatement node)
      throws EvalException, InterruptedException {
    try {
      if (node.isAugmented()) {
        execAugmentedAssignment(fr, node);
      } else {
        Object rvalue = eval(fr, node.getRHS());
        assign(fr, node.getLHS(), rvalue);
      }
    } catch (EvalException ex) {
      fr.setErrorLocation(node.getOperatorLocation());
      throw ex;
    }
  }

  private static TokenKind execFor(StarlarkThread.Frame fr, ForStatement node)
      throws EvalException, InterruptedException {
    Object o = eval(fr, node.getCollection());
    Iterable<?> seq = Starlark.toIterable(o);
    EvalUtils.addIterator(o);
    try {
      for (Object it : seq) {
        assign(fr, node.getVars(), it);

        switch (execStatements(fr, node.getBody(), /*indented=*/ true)) {
          case PASS:
          case CONTINUE:
            // Stay in loop.
            fr.thread.checkInterrupt();
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
      fr.setErrorLocation(node.getStartLocation());
      throw ex;
    } finally {
      EvalUtils.removeIterator(o);
    }
    return TokenKind.PASS;
  }

  private static void execDef(StarlarkThread.Frame fr, DefStatement node)
      throws EvalException, InterruptedException {
    Resolver.Function rfn = node.getResolvedFunction();

    // Evaluate default value expressions of optional parameters.
    // We use MANDATORY to indicate a required parameter
    // (not null, because defaults must be a legal tuple value, as
    // it will be constructed by the code emitted by the compiler).
    // As an optimization, we omit the prefix of MANDATORY parameters.
    Object[] defaults = null;
    int nparams =
        rfn.getParameters().size() - (rfn.hasKwargs() ? 1 : 0) - (rfn.hasVarargs() ? 1 : 0);
    for (int i = 0; i < nparams; i++) {
      Expression expr = rfn.getParameters().get(i).getDefaultValue();
      if (expr == null && defaults == null) {
        continue; // skip prefix of required parameters
      }
      if (defaults == null) {
        defaults = new Object[nparams - i];
      }
      defaults[i - (nparams - defaults.length)] =
          expr == null ? StarlarkFunction.MANDATORY : eval(fr, expr);
    }
    if (defaults == null) {
      defaults = EMPTY;
    }

    assignIdentifier(
        fr,
        node.getIdentifier(),
        new StarlarkFunction(rfn, Tuple.wrap(defaults), fn(fr).getModule()));
  }

  private static TokenKind execIf(StarlarkThread.Frame fr, IfStatement node)
      throws EvalException, InterruptedException {
    boolean cond = Starlark.truth(eval(fr, node.getCondition()));
    if (cond) {
      return execStatements(fr, node.getThenBlock(), /*indented=*/ true);
    } else if (node.getElseBlock() != null) {
      return execStatements(fr, node.getElseBlock(), /*indented=*/ true);
    }
    return TokenKind.PASS;
  }

  private static void execLoad(StarlarkThread.Frame fr, LoadStatement node) throws EvalException {
    // Has the application defined a behavior for load statements in this thread?
    StarlarkThread.Loader loader = fr.thread.getLoader();
    if (loader == null) {
      fr.setErrorLocation(node.getStartLocation());
      throw Starlark.errorf("load statements may not be executed in this thread");
    }

    // Load module.
    String moduleName = node.getImport().getValue();
    Module module = loader.load(moduleName);
    if (module == null) {
      fr.setErrorLocation(node.getStartLocation());
      throw Starlark.errorf(
          "file '%s' was not correctly loaded. Make sure the 'load' statement appears in the"
              + " global scope in your file",
          moduleName);
    }
    Map<String, Object> globals = module.getExportedGlobals();

    for (LoadStatement.Binding binding : node.getBindings()) {
      // Extract symbol.
      Identifier orig = binding.getOriginalName();
      Object value = globals.get(orig.getName());
      if (value == null) {
        fr.setErrorLocation(orig.getStartLocation());
        throw Starlark.errorf(
            "file '%s' does not contain symbol '%s'%s",
            moduleName, orig.getName(), SpellChecker.didYouMean(orig.getName(), globals.keySet()));
      }

      // Define module-local variable.
      // TODO(adonovan): eventually the default behavior should be that
      // loads bind file-locally. Either way, the resolver should designate
      // the proper scope of binding.getLocalName() and this should become
      // simply assign(binding.getLocalName(), value).
      // Currently, we update the module but not module.exportedGlobals;
      // changing it to fr.locals.put breaks a test. TODO(adonovan): find out why.
      fn(fr).getModule().setGlobal(binding.getLocalName().getName(), value);
    }
  }

  private static TokenKind execReturn(StarlarkThread.Frame fr, ReturnStatement node)
      throws EvalException, InterruptedException {
    Expression result = node.getResult();
    if (result != null) {
      fr.result = eval(fr, result);
    }
    return TokenKind.RETURN;
  }

  private static TokenKind exec(StarlarkThread.Frame fr, Statement st)
      throws EvalException, InterruptedException {
    if (fr.dbg != null) {
      Location loc = st.getStartLocation(); // not very precise
      fr.setLocation(loc);
      fr.dbg.before(fr.thread, loc); // location is now redundant since it's in the thread
    }

    if (++fr.thread.steps >= fr.thread.stepLimit) {
      throw new EvalException("Starlark computation cancelled: too many steps");
    }

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
   * to the given expression. Might not set the frame location on error.
   */
  private static void assign(StarlarkThread.Frame fr, Expression lhs, Object value)
      throws EvalException, InterruptedException {
    if (lhs instanceof Identifier) {
      // x = ...
      assignIdentifier(fr, (Identifier) lhs, value);

    } else if (lhs instanceof IndexExpression) {
      // x[i] = ...
      Object object = eval(fr, ((IndexExpression) lhs).getObject());
      Object key = eval(fr, ((IndexExpression) lhs).getKey());
      EvalUtils.setIndex(object, key, value);

    } else if (lhs instanceof ListExpression) {
      // a, b, c = ...
      ListExpression list = (ListExpression) lhs;
      assignSequence(fr, list.getElements(), value);

    } else if (lhs instanceof DotExpression) {
      // x.f = ...
      DotExpression dot = (DotExpression) lhs;
      Object object = eval(fr, dot.getObject());
      String field = dot.getField().getName();
      try {
        EvalUtils.setField(object, field, value);
      } catch (EvalException ex) {
        fr.setErrorLocation(dot.getDotLocation());
        throw ex;
      }
    } else {
      // Not possible for resolved ASTs.
      throw Starlark.errorf("cannot assign to '%s'", lhs);
    }
  }

  private static void assignIdentifier(StarlarkThread.Frame fr, Identifier id, Object value)
      throws EvalException {
    Resolver.Binding bind = id.getBinding();
    // Legacy hack for incomplete identifier resolution.
    // In a <toplevel> function, assignments to unresolved identifiers
    // update the module, except for load statements and comprehensions,
    // which should both be file-local.
    // Load statements don't yet use assignIdentifier,
    // so we need consider only comprehensions.
    // In effect, we do the missing resolution using fr.compcount.
    Resolver.Scope scope;
    if (bind == null) {
      scope =
          fn(fr).isToplevel() && fr.compcount == 0
              ? Resolver.Scope.GLOBAL //
              : Resolver.Scope.LOCAL;
    } else {
      scope = bind.getScope();
    }

    String name = id.getName();
    switch (scope) {
      case LOCAL:
        fr.locals.put(name, value);
        break;
      case GLOBAL:
        // Updates a module binding and sets its 'exported' flag.
        // (Only load bindings are not exported.
        // But exportedGlobals does at run time what should be done in the resolver.)
        Module module = fn(fr).getModule();
        module.setGlobal(name, value);
        module.exportedGlobals.add(name);
        break;
      default:
        throw new IllegalStateException(scope.toString());
    }
  }

  /**
   * Recursively assigns an iterable value to a non-empty sequence of assignable expressions. Might
   * not set frame location on error.
   */
  private static void assignSequence(StarlarkThread.Frame fr, List<Expression> lhs, Object x)
      throws EvalException, InterruptedException {
    // TODO(adonovan): lock/unlock rhs during iteration so that
    // assignments fail when the left side aliases the right,
    // which is a tricky case in Python assignment semantics.
    int nrhs = Starlark.len(x);
    int nlhs = lhs.size();
    if (nrhs < 0 || x instanceof String) { // strings are not iterable
      throw Starlark.errorf(
          "got '%s' in sequence assignment (want %d-element sequence)", Starlark.type(x), nlhs);
    }
    Iterable<?> rhs = Starlark.toIterable(x);
    if (nlhs != nrhs) {
      throw Starlark.errorf(
          "too %s values to unpack (got %d, want %d)", nrhs < nlhs ? "few" : "many", nrhs, nlhs);
    }
    int i = 0;
    for (Object item : rhs) {
      assign(fr, lhs.get(i), item);
      i++;
    }
  }

  // Might not set frame location on error.
  private static void execAugmentedAssignment(StarlarkThread.Frame fr, AssignmentStatement stmt)
      throws EvalException, InterruptedException {
    Expression lhs = stmt.getLHS();
    TokenKind op = stmt.getOperator();
    Expression rhs = stmt.getRHS();

    if (lhs instanceof Identifier) {
      Object x = eval(fr, lhs);
      Object y = eval(fr, rhs);
      Object z = inplaceBinaryOp(fr, op, x, y);
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
      Object z = inplaceBinaryOp(fr, op, x, y);
      try {
        EvalUtils.setIndex(object, key, z);
      } catch (EvalException ex) {
        fr.setErrorLocation(stmt.getOperatorLocation());
        throw ex;
      }

    } else {
      // Not possible for resolved ASTs.
      fr.setErrorLocation(stmt.getOperatorLocation());
      throw Starlark.errorf("cannot perform augmented assignment on '%s'", lhs);
    }
  }

  private static Object inplaceBinaryOp(StarlarkThread.Frame fr, TokenKind op, Object x, Object y)
      throws EvalException {
    // list += iterable  behaves like  list.extend(iterable)
    // TODO(b/141263526): following Python, allow list+=iterable (but not list+iterable).
    if (op == TokenKind.PLUS && x instanceof StarlarkList && y instanceof StarlarkList) {
      StarlarkList<?> list = (StarlarkList) x;
      list.extend(y);
      return list;
    }
    return EvalUtils.binaryOp(op, x, y, fr.thread.getSemantics(), fr.thread.mutability());
  }

  // ---- expressions ----

  private static Object eval(StarlarkThread.Frame fr, Expression expr)
      throws EvalException, InterruptedException {
    if (++fr.thread.steps >= fr.thread.stepLimit) {
      throw new EvalException("Starlark computation cancelled: too many steps");
    }

    // The switch cases have been split into separate functions
    // to reduce the stack usage during recursion, which is
    // especially important in practice for deeply nested a+...+z
    // expressions; see b/153764542.
    switch (expr.kind()) {
      case BINARY_OPERATOR:
        return evalBinaryOperator(fr, (BinaryOperatorExpression) expr);
      case COMPREHENSION:
        return evalComprehension(fr, (Comprehension) expr);
      case CONDITIONAL:
        return evalConditional(fr, (ConditionalExpression) expr);
      case DICT_EXPR:
        return evalDict(fr, (DictExpression) expr);
      case DOT:
        return evalDot(fr, (DotExpression) expr);
      case CALL:
        return evalCall(fr, (CallExpression) expr);
      case IDENTIFIER:
        return evalIdentifier(fr, (Identifier) expr);
      case INDEX:
        return evalIndex(fr, (IndexExpression) expr);
      case INT_LITERAL:
        // TODO(adonovan): opt: avoid allocation by saving
        // the StarlarkInt in the IntLiteral (a temporary hack
        // until we use a compiled representation).
        Number n = ((IntLiteral) expr).getValue();
        if (n instanceof Integer) {
          return StarlarkInt.of((Integer) n);
        } else if (n instanceof Long) {
          return StarlarkInt.of((Long) n);
        } else {
          return StarlarkInt.of((BigInteger) n);
        }
      case FLOAT_LITERAL:
        return StarlarkFloat.of(((FloatLiteral) expr).getValue());
      case LIST_EXPR:
        return evalList(fr, (ListExpression) expr);
      case SLICE:
        return evalSlice(fr, (SliceExpression) expr);
      case STRING_LITERAL:
        return ((StringLiteral) expr).getValue();
      case UNARY_OPERATOR:
        return evalUnaryOperator(fr, (UnaryOperatorExpression) expr);
    }
    throw new IllegalArgumentException("unexpected expression: " + expr.kind());
  }

  private static Object evalBinaryOperator(StarlarkThread.Frame fr, BinaryOperatorExpression binop)
      throws EvalException, InterruptedException {
    Object x = eval(fr, binop.getX());
    // AND and OR require short-circuit evaluation.
    switch (binop.getOperator()) {
      case AND:
        return Starlark.truth(x) ? eval(fr, binop.getY()) : x;
      case OR:
        return Starlark.truth(x) ? x : eval(fr, binop.getY());
      default:
        Object y = eval(fr, binop.getY());
        try {
          return EvalUtils.binaryOp(
              binop.getOperator(), x, y, fr.thread.getSemantics(), fr.thread.mutability());
        } catch (EvalException ex) {
          fr.setErrorLocation(binop.getOperatorLocation());
          throw ex;
        }
    }
  }

  private static Object evalConditional(StarlarkThread.Frame fr, ConditionalExpression cond)
      throws EvalException, InterruptedException {
    Object v = eval(fr, cond.getCondition());
    return eval(fr, Starlark.truth(v) ? cond.getThenCase() : cond.getElseCase());
  }

  private static Object evalDict(StarlarkThread.Frame fr, DictExpression dictexpr)
      throws EvalException, InterruptedException {
    Dict<Object, Object> dict = Dict.of(fr.thread.mutability());
    for (DictExpression.Entry entry : dictexpr.getEntries()) {
      Object k = eval(fr, entry.getKey());
      Object v = eval(fr, entry.getValue());
      int before = dict.size();
      try {
        dict.putEntry(k, v);
      } catch (EvalException ex) {
        fr.setErrorLocation(entry.getColonLocation());
        throw ex;
      }
      if (dict.size() == before) {
        fr.setErrorLocation(entry.getColonLocation());
        throw Starlark.errorf("dictionary expression has duplicate key: %s", Starlark.repr(k));
      }
    }
    return dict;
  }

  private static Object evalDot(StarlarkThread.Frame fr, DotExpression dot)
      throws EvalException, InterruptedException {
    Object object = eval(fr, dot.getObject());
    String name = dot.getField().getName();
    try {
      return Starlark.getattr(
          fr.thread.mutability(), fr.thread.getSemantics(), object, name, /*defaultValue=*/ null);
    } catch (EvalException ex) {
      fr.setErrorLocation(dot.getDotLocation());
      throw ex;
    }
  }

  private static Object evalCall(StarlarkThread.Frame fr, CallExpression call)
      throws EvalException, InterruptedException {
    fr.thread.checkInterrupt();

    Object fn = eval(fr, call.getFunction());

    // Starlark arguments are ordered: positionals < keywords < *args < **kwargs.
    //
    // This is stricter than Python2, which doesn't constrain keywords wrt *args,
    // but this ensures that the effects of evaluation of Starlark arguments occur
    // in source order.
    //
    // Starlark does not support Python3's multiple *args and **kwargs
    // nor freer ordering, such as f(a, *list, *list, **dict, **dict, b=1).
    // Supporting it would complicate a compiler, and produce effects out of order.
    // Also, Python's argument ordering rules are complex and the errors sometimes cryptic.

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
        fr.setErrorLocation(star.getStartLocation());
        throw Starlark.errorf("argument after * must be an iterable, not %s", Starlark.type(value));
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
        fr.setErrorLocation(starstar.getStartLocation());
        throw Starlark.errorf("argument after ** must be a dict, not %s", Starlark.type(value));
      }
      Dict<?, ?> kwargs = (Dict<?, ?>) value;
      int j = named.length;
      named = Arrays.copyOf(named, j + 2 * kwargs.size());
      for (Map.Entry<?, ?> e : kwargs.entrySet()) {
        if (!(e.getKey() instanceof String)) {
          fr.setErrorLocation(starstar.getStartLocation());
          throw Starlark.errorf("keywords must be strings, not %s", Starlark.type(e.getKey()));
        }
        named[j++] = e.getKey();
        named[j++] = e.getValue();
      }
    }

    Location loc = call.getLparenLocation(); // (Location is prematerialized)
    fr.setLocation(loc);
    try {
      return Starlark.fastcall(fr.thread, fn, positional, named);
    } catch (EvalException ex) {
      fr.setErrorLocation(loc);
      throw ex;
    }
  }

  private static Object evalIdentifier(StarlarkThread.Frame fr, Identifier id)
      throws EvalException, InterruptedException {
    String name = id.getName();
    Resolver.Binding bind = id.getBinding();
    if (bind == null) {
      // Legacy behavior, to be removed.
      Object result = fr.locals.get(name);
      if (result != null) {
        return result;
      }
      result = fn(fr).getModule().get(name);
      if (result != null) {
        return result;
      }

      // Assuming resolution was successfully applied before execution
      // (which is not yet true for copybara, but will be soon),
      // then the identifier must have been resolved but the
      // resolution was not annotated onto the syntax tree---because
      // it's a BUILD file that may share trees with the prelude.
      // So this error does not mean "undefined variable" (morally a
      // static error), but "variable was (dynamically) referenced
      // before being bound", as in 'print(x); x=1'.
      fr.setErrorLocation(id.getStartLocation());
      throw Starlark.errorf("variable '%s' is referenced before assignment", name);
    }

    Object result;
    switch (bind.getScope()) {
      case LOCAL:
        result = fr.locals.get(name);
        break;
      case GLOBAL:
        result = fn(fr).getModule().getGlobal(name);
        break;
      case PREDECLARED:
        // TODO(adonovan): call getPredeclared
        result = fn(fr).getModule().get(name);
        break;
      default:
        throw new IllegalStateException(bind.toString());
    }
    if (result == null) {
      // Since Scope was set, we know that the local/global variable is defined,
      // but its assignment was not yet executed.
      fr.setErrorLocation(id.getStartLocation());
      throw Starlark.errorf(
          "%s variable '%s' is referenced before assignment.", bind.getScope(), name);
    }
    return result;
  }

  private static Object evalIndex(StarlarkThread.Frame fr, IndexExpression index)
      throws EvalException, InterruptedException {
    Object object = eval(fr, index.getObject());
    Object key = eval(fr, index.getKey());
    try {
      return EvalUtils.index(fr.thread.mutability(), fr.thread.getSemantics(), object, key);
    } catch (EvalException ex) {
      fr.setErrorLocation(index.getLbracketLocation());
      throw ex;
    }
  }

  private static Object evalList(StarlarkThread.Frame fr, ListExpression expr)
      throws EvalException, InterruptedException {
    int n = expr.getElements().size();
    Object[] array = new Object[n];
    for (int i = 0; i < n; i++) {
      array[i] = eval(fr, expr.getElements().get(i));
    }
    return expr.isTuple() ? Tuple.wrap(array) : StarlarkList.wrap(fr.thread.mutability(), array);
  }

  private static Object evalSlice(StarlarkThread.Frame fr, SliceExpression slice)
      throws EvalException, InterruptedException {
    Object x = eval(fr, slice.getObject());
    Object start = slice.getStart() == null ? Starlark.NONE : eval(fr, slice.getStart());
    Object stop = slice.getStop() == null ? Starlark.NONE : eval(fr, slice.getStop());
    Object step = slice.getStep() == null ? Starlark.NONE : eval(fr, slice.getStep());
    try {
      return Starlark.slice(fr.thread.mutability(), x, start, stop, step);
    } catch (EvalException ex) {
      fr.setErrorLocation(slice.getLbracketLocation());
      throw ex;
    }
  }

  private static Object evalUnaryOperator(StarlarkThread.Frame fr, UnaryOperatorExpression unop)
      throws EvalException, InterruptedException {
    Object x = eval(fr, unop.getX());
    try {
      return EvalUtils.unaryOp(unop.getOperator(), x);
    } catch (EvalException ex) {
      fr.setErrorLocation(unop.getStartLocation());
      throw ex;
    }
  }

  private static Object evalComprehension(StarlarkThread.Frame fr, Comprehension comp)
      throws EvalException, InterruptedException {
    final Dict<Object, Object> dict = comp.isDict() ? Dict.of(fr.thread.mutability()) : null;
    final StarlarkList<Object> list =
        comp.isDict() ? null : StarlarkList.of(fr.thread.mutability());

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
        fr.thread.checkInterrupt();

        // recursive case: one or more clauses
        if (index < comp.getClauses().size()) {
          Comprehension.Clause clause = comp.getClauses().get(index);
          if (clause instanceof Comprehension.For) {
            Comprehension.For forClause = (Comprehension.For) clause;

            Object iterable = eval(fr, forClause.getIterable());
            Iterable<?> seq = Starlark.toIterable(iterable);
            EvalUtils.addIterator(iterable);
            try {
              for (Object elem : seq) {
                assign(fr, forClause.getVars(), elem);
                execClauses(index + 1);
              }
            } catch (EvalException ex) {
              fr.setErrorLocation(forClause.getStartLocation());
              throw ex;
            } finally {
              EvalUtils.removeIterator(iterable);
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
          try {
            Starlark.checkHashable(k);
            Object v = eval(fr, body.getValue());
            dict.putEntry(k, v);
          } catch (EvalException ex) {
            fr.setErrorLocation(body.getColonLocation());
            throw ex;
          }
        } else {
          list.addElement(eval(fr, ((Expression) comp.getBody())));
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

    return comp.isDict() ? dict : list;
  }

  private static final Object[] EMPTY = {};
}
