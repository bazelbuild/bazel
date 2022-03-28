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

package net.starlark.java.eval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

/**
 *
 */
public class SortKeyHelper {

  private final List<KeyCallbackComparable> items;
  private final StarlarkThread thread;

  private SortKeyHelper(List<KeyCallbackComparable> items, StarlarkThread thread) {
    this.items = items;
    this.thread = thread;
  }

  private static class KeyCallbackComparable {
    public final Object key;
    public final Object value;

    public KeyCallbackComparable(Object key, Object value) {
      this.key = key;
      this.value = value;
    }

    public static KeyCallbackComparable withValueAndKey(Object value, Object key, StarlarkThread thread) throws EvalException, InterruptedException {
      if (!(key instanceof StarlarkCallable) && key != Starlark.NONE) {
        throw Starlark.errorf("for key, got %s, want callable", Starlark.type(key));
      }
      return new KeyCallbackComparable(
        key == Starlark.NONE ? value : Starlark.fastcall(thread, key, new Object[]{value}, new Object[]{}),
        value
      );
    }
  }

  private static class KeyComparator implements Comparator<KeyCallbackComparable> {

    private final Comparator<Object> order;

    public KeyComparator(Comparator<Object> order) {
      this.order = order;
    }

    @Override
    public int compare(KeyCallbackComparable x, KeyCallbackComparable y) {
      return order.compare(x.key, y.key);
    }
  }

  public static SortKeyHelper withItemsAndKey(StarlarkIterable<?> items, Object key, StarlarkThread thread) throws EvalException, InterruptedException {
    List<KeyCallbackComparable> list = new ArrayList<>();
    for (Object item : items) {
      list.add(KeyCallbackComparable.withValueAndKey(item, key, thread));
    }
    return new SortKeyHelper(list, thread);
  }

  public static SortKeyHelper withItemsAndKey(Sequence<?> items, Object key, StarlarkThread thread) throws EvalException, InterruptedException {
    return withItemsAndKey(unwrapListFromExtraPositional(items), key, thread);
  }

  public Object min() throws EvalException {
    try {
      checkNotEmpty(items);
      return Collections.min(this.items, new KeyComparator(Starlark.ORDERING)).value;
    } catch (ClassCastException e) {
      throw new EvalException(e.getMessage());
    }
  }

  public Object max() throws EvalException {
    try {
      checkNotEmpty(items);
      return Collections.max(this.items, new KeyComparator(Starlark.ORDERING)).value;
    } catch (ClassCastException e) {
      throw new EvalException(e.getMessage());
    }
  }

  public StarlarkIterable<?> sorted(boolean reversed) throws EvalException {
    try {
      return StarlarkList.wrap(
        thread.mutability(),
        this.items.stream()
          .sorted(new KeyComparator(reversed ? Starlark.ORDERING.reversed() : Starlark.ORDERING))
          .map(item -> item.value)
          .toArray()
      );
    } catch (ClassCastException e) {
      throw new EvalException(e);
    }
  }

  private static StarlarkIterable<?> unwrapListFromExtraPositional(Sequence<?> args) throws EvalException {
    checkNotEmpty(args);
    if (args.size() > 1) {
      return args;
    }
    Object item = args.get(0);
    try {
      return ((StarlarkIterable<?>) item);
    } catch (ClassCastException e) {
      throw Starlark.errorf("type '%s' is not iterable", Starlark.type(item));
    }
  }

  private static void checkNotEmpty(List<?> it) throws EvalException {
    if (it.isEmpty()) {
      throw new EvalException("expected at least one item");
    }
  }
}
