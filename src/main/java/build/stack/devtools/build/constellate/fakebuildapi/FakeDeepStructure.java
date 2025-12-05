// Copyright 2021 The Bazel Authors. All rights reserved.
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

package build.stack.devtools.build.constellate.fakebuildapi;

import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import javax.annotation.Nullable;
import net.starlark.java.annot.StarlarkBuiltin;
import net.starlark.java.eval.Dict;
import net.starlark.java.eval.EvalException;
import net.starlark.java.eval.Printer;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.HasBinary;
import net.starlark.java.eval.Mutability;
import net.starlark.java.eval.Sequence;
import net.starlark.java.eval.StarlarkCallable;
import net.starlark.java.eval.StarlarkIndexable;
import net.starlark.java.eval.StarlarkList;
import net.starlark.java.eval.StarlarkSemantics;
import net.starlark.java.eval.StarlarkThread;
import net.starlark.java.eval.Structure;
import net.starlark.java.eval.Tuple;
import net.starlark.java.syntax.TokenKind;
import java.util.Iterator;

/**
 * A fake Starlark structure that returns itself for any field that it's asked for and that can be
 * called. Implements both HasBinary for binary operations and Sequence for type compatibility.
 */
@StarlarkBuiltin(name = "FakeDeepStructure", documented = false)
public class FakeDeepStructure extends FakeProviderApi implements Structure, StarlarkCallable, StarlarkIndexable, HasBinary, Sequence<Object> {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final String fullName;

  private FakeDeepStructure(@Nullable String name, String fullName) {
    super(name);
    this.fullName = fullName;
  }

  /** Creates a new fake deep structure with the given name. */
  public static FakeDeepStructure create(String name) {
    return new FakeDeepStructure(name, name);
  }

  @Nullable
  @Override
  public Object getValue(String name) throws EvalException {
    logger.atFine().log("FakeDeepStructure.getValue: %s.%s", fullName, name);
    return new FakeDeepStructure(name, fullName + "." + name);
  }

  @Override
  public ImmutableCollection<String> getFieldNames() {
    return ImmutableList.of();
  }

  @Nullable
  @Override
  public String getErrorMessageForUnknownField(String field) {
    return null;
  }

  @Override
  public Object call(StarlarkThread thread, Tuple args, Dict<String, Object> kwargs) {
    logger.atFine().log("FakeAPI call: %s(%s, %s) at %s", fullName, args, kwargs, thread.getCallerLocation());
    // Return another FakeDeepStructure so it can be chained or used in any context
    return new FakeDeepStructure(getName() + "()", fullName + "()");
  }

  @Override
  public void repr(Printer printer) {
    printer.append("<fake object ").append(fullName).append(">");
  }

  @Override
  public Object getIndex(StarlarkSemantics semantics, Object key) throws EvalException {
    // String indexName = "[" + key + "]";
    return Starlark.NONE;
    // return Sequence
    // return new FakeDeepStructure(getName() + key, fullName + key);
  }

  @Override
  public boolean containsKey(StarlarkSemantics semantics, Object key) throws EvalException {
    return false;
  }

  @Override
  public Object binaryOp(TokenKind op, Object that, boolean thisLeft) throws EvalException {
    // Handle binary operations with FakeDeepStructure
    // For PLUS (concatenation), return the other operand unchanged
    // This allows list + FakeDeepStructure to work transparently
    if (op == TokenKind.PLUS) {
      logger.atFine().log("FakeDeepStructure.binaryOp: %s %s %s (thisLeft=%s)",
          thisLeft ? fullName : that, op, thisLeft ? that : fullName, thisLeft);
      // Return the other operand (the list) unchanged
      return that;
    }
    // For other operations, return null to let default handling occur
    return null;
  }

  // Sequence interface methods - behave like an empty list
  @Override
  public Iterator<Object> iterator() {
    return ImmutableList.of().iterator();
  }

  @Override
  public boolean isEmpty() {
    return true;
  }

  @Override
  public int size() {
    return 0;
  }

  @Override
  public Object get(int index) {
    return new FakeDeepStructure("[" + index + "]", fullName + "[" + index + "]");
  }

  @Override
  public Sequence<Object> getSlice(Mutability mu, int start, int stop, int step) {
    return StarlarkList.empty();
  }

  @Override
  public ImmutableList<Object> subList(int fromIndex, int toIndex) {
    return ImmutableList.of();
  }

  @Override
  public java.util.ListIterator<Object> listIterator() {
    return ImmutableList.of().listIterator();
  }

  @Override
  public java.util.ListIterator<Object> listIterator(int index) {
    return ImmutableList.of().listIterator(index);
  }

  @Override
  public boolean contains(Object o) {
    return false;
  }

  @Override
  public Object[] toArray() {
    return new Object[0];
  }

  @Override
  public <T> T[] toArray(T[] a) {
    if (a.length > 0) {
      a[0] = null;
    }
    return a;
  }

  @Override
  public boolean add(Object e) {
    throw new UnsupportedOperationException("FakeDeepStructure is immutable");
  }

  @Override
  public boolean remove(Object o) {
    throw new UnsupportedOperationException("FakeDeepStructure is immutable");
  }

  @Override
  public boolean containsAll(java.util.Collection<?> c) {
    return c.isEmpty();
  }

  @Override
  public boolean addAll(java.util.Collection<?> c) {
    throw new UnsupportedOperationException("FakeDeepStructure is immutable");
  }

  @Override
  public boolean addAll(int index, java.util.Collection<?> c) {
    throw new UnsupportedOperationException("FakeDeepStructure is immutable");
  }

  @Override
  public boolean removeAll(java.util.Collection<?> c) {
    throw new UnsupportedOperationException("FakeDeepStructure is immutable");
  }

  @Override
  public boolean retainAll(java.util.Collection<?> c) {
    throw new UnsupportedOperationException("FakeDeepStructure is immutable");
  }

  @Override
  public void clear() {
    throw new UnsupportedOperationException("FakeDeepStructure is immutable");
  }

  @Override
  public Object set(int index, Object element) {
    throw new UnsupportedOperationException("FakeDeepStructure is immutable");
  }

  @Override
  public void add(int index, Object element) {
    throw new UnsupportedOperationException("FakeDeepStructure is immutable");
  }

  @Override
  public Object remove(int index) {
    throw new UnsupportedOperationException("FakeDeepStructure is immutable");
  }

  @Override
  public int indexOf(Object o) {
    return -1;
  }

  @Override
  public int lastIndexOf(Object o) {
    return -1;
  }

}
