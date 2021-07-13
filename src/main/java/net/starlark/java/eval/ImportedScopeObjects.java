package net.starlark.java.eval;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

import java.util.stream.IntStream;

/** Map from name to value with index by name. */
class ImportedScopeObjects {

  private final ImmutableMap<String, Object> byName;
  private final ImmutableList<Object> byIndex;
  private final ImmutableMap<String, Integer> indexByName;

  private ImportedScopeObjects(ImmutableMap<String, Object> byName) {
    this.byName = byName;

    ImmutableList<String> names = byName.keySet().asList();
    this.indexByName =
        IntStream.range(0, names.size())
            .boxed()
            .collect(ImmutableMap.toImmutableMap(names::get, i -> i));

    // This is consistent with `indexByName` because `ImmutableMap` is ordered.
    this.byIndex = byName.values().asList();
  }

  /** Create a scope object from given value map by name. */
  public static ImportedScopeObjects create(ImmutableMap<String, Object> valuesByName) {
    return new ImportedScopeObjects(valuesByName);
  }

  /** Get value index by name or {@code -1} if name is undefined. */
  public int indexByName(String name) {
    Integer index = indexByName.get(name);
    return index != null ? index : -1;
  }

  /** Get a value by index. */
  public Object valueByIndex(int index) {
    return byIndex.get(index);
  }
}
