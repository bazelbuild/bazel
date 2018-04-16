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

package com.google.devtools.build.lib.skyframe.serialization;

import com.google.auto.value.AutoValue;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Encapsulation of value constants for serialization. Care is taken to avoid potentially expensive
 * hash code/equality calculations: we store values in a two-level map, keyed by class. That way,
 * only objects with the same class as a constant will ever have {@link Object#hashCode} or {@link
 * Object#equals}called on them.
 *
 * <p>An even more elaborate scheme is used for non-empty {@link Collection} objects, which may
 * potentially have hash code/equality-hostile elements. We key collections based on their class,
 * the class of the items inside them, and their size. Only if all three match do we use normal hash
 * code/equals- based checking. Since the class of items in the {@link Collection} is relevant,
 * {@link Collection}s stored here must contain at least one non-null element, and the class of the
 * first non-null element is taken as the class of all items in the collection.
 */
public class ValueConstants {
  private final int constantsStartTag;
  /**
   * Map from class of constant to map of constant to index. First level ensures we only check
   * objects for constant-ness if they are of an appropriate class, and avoids collisions between
   * objects that may be of different classes but happen to compare identical (like Lists).
   */
  private final ImmutableMap<Class<?>, ImmutableMap<Object, Integer>> simpleConstantsMap;
  /**
   * Map from {@link CollectionInfo} to map of constant collection to index. First level ensures we
   * only check objects for constant-ness if they are collections of the same class, and containing
   * the same class, and with the same size.
   */
  private final ImmutableMap<CollectionInfo, ImmutableMap<Collection<?>, Integer>>
      collectionConstantsMap;

  private final ImmutableList<Object> constants;

  private ValueConstants(
      int constantsStartTag,
      ImmutableMap<Class<?>, ImmutableMap<Object, Integer>> simpleConstantsMap,
      ImmutableMap<CollectionInfo, ImmutableMap<Collection<?>, Integer>> collectionConstantsMap,
      ImmutableList<Object> constants) {
    this.constantsStartTag = constantsStartTag;
    this.simpleConstantsMap = simpleConstantsMap;
    this.collectionConstantsMap = collectionConstantsMap;
    this.constants = constants;
  }

  int getNextTag() {
    return constantsStartTag + constants.size();
  }

  int size() {
    return constants.size();
  }

  @Nullable
  Integer maybeGetTagForConstant(Object object) {
    Collection<?> collection = castIfNonEmptyCollection(object);
    if (collection != null) {
      CollectionInfo collectionInfo = CollectionInfo.makeCollectionInfo(collection);
      if (collectionInfo == null) {
        return null;
      }
      ImmutableMap<Collection<?>, Integer> map = collectionConstantsMap.get(collectionInfo);
      return map != null ? map.get(collection) : null;
    }
    ImmutableMap<Object, Integer> map = simpleConstantsMap.get(object.getClass());
    return map != null ? map.get(object) : null;
  }

  @Nullable
  Object maybeGetConstantByTag(int tag) {
    tag = tag - constantsStartTag;
    return 0 <= tag && tag < constants.size() ? constants.get(tag) : null;
  }

  Builder toBuilder() {
    ArrayList<Object> simpleConstants = new ArrayList<>(constants.size());
    ArrayList<CollectionAndCollectionInfo> collectionConstants = new ArrayList<>(constants.size());
    for (Object constant : constants) {
      Collection<?> collection = castIfNonEmptyCollection(constant);
      if (collection == null) {
        simpleConstants.add(constant);
      } else {
        CollectionInfo collectionInfo =
            Preconditions.checkNotNull(CollectionInfo.makeCollectionInfo(collection), collection);
        Map<Collection<?>, Integer> sanityCheckMap =
            Preconditions.checkNotNull(
                collectionConstantsMap.get(collectionInfo), "%s %s", collection, collectionInfo);
        Preconditions.checkState(
            sanityCheckMap.containsKey(collection),
            "%s %s %s",
            collection,
            collectionInfo,
            sanityCheckMap);
        collectionConstants.add(CollectionAndCollectionInfo.of(collection, collectionInfo));
      }
    }
    return new Builder(collectionConstants, simpleConstants);
  }

  @SuppressWarnings({"unchecked", "rawtypes"})
  private static Class<? extends Collection<?>> castCollectionClass(
      Class<? extends Collection> collectionClass) {
    return (Class<? extends Collection<?>>) collectionClass;
  }

  @Nullable
  private static Collection<?> castIfNonEmptyCollection(Object object) {
    if (object instanceof Collection<?>) {
      Collection<?> collection = (Collection<?>) object;
      if (!collection.isEmpty()) {
        return collection;
      }
    }
    return null;
  }

  /** Builder for {@link ValueConstants}. */
  public static class Builder {
    private boolean immutable = false;
    private final List<CollectionAndCollectionInfo> collectionConstants;
    private final List<Object> simpleConstants;

    private Builder(
        List<CollectionAndCollectionInfo> collectionConstants, List<Object> simpleConstants) {
      this.collectionConstants = collectionConstants;
      this.simpleConstants = simpleConstants;
    }

    public Builder() {
      this(new ArrayList<>(), new ArrayList<>());
    }

    public Builder copy() {
      return new Builder(collectionConstants, simpleConstants);
    }

    public Builder makeImmutable() {
      immutable = true;
      return this;
    }

    public Builder merge(Builder other) {
      Preconditions.checkState(!immutable);
      collectionConstants.addAll(other.collectionConstants);
      simpleConstants.addAll(other.simpleConstants);
      return this;
    }

    /**
     * Adds a constant collection value. {@code collection} must not be empty, and {@code
     * itemClassForSanityCheck} must be the type of all elements in {@code collection}. Similar to
     * {@link #addSimpleConstant}, but ensures that equality will only be checked for other {@link
     * Collection}s whose items have the same type as {@code collection}, namely {@code
     * itemClassForSanityCheck}.
     *
     * <p>{@code itemClassForSanityCheck} is only used to sanity check that the computed class
     * inside {@code collection} (coming from the first non-null element) agrees with the caller's
     * notion of the class of items inside the collection.
     */
    public <T> Builder addCollectionConstant(
        Collection<T> collection, Class<T> itemClassForSanityCheck) {
      Preconditions.checkState(!immutable);
      CollectionInfo collectionInfo =
          Preconditions.checkNotNull(
              CollectionInfo.makeCollectionInfo(collection),
              "CollectionInfo couldn't be calculated for value constant %s",
              collection);
      Preconditions.checkState(
          collectionInfo.getItemClass().equals(itemClassForSanityCheck),
          "Item class mismatch for %s: %s and %s (%s)",
          collection,
          itemClassForSanityCheck,
          collectionInfo.getItemClass(),
          collectionInfo);
      collectionConstants.add(CollectionAndCollectionInfo.of(collection, collectionInfo));
      return this;
    }

    /**
     * Adds a constant value. Any value encountered during serialization which has the same class as
     * {@code object} and {@link Object#equals} {@code object} will be replaced by {@code object}
     * upon deserialization. These objects should therefore be indistinguishable, and unequal
     * objects should quickly compare unequal (it is ok for equal objects to be relatively expensive
     * to compare equal, if that is still less expensive than the cost of serializing the object).
     * Short {@link String} objects are ideal for value constants.
     *
     * <p>Empty collections should be added here, while non-empty collections should be added using
     * {@link #addCollectionConstant}.
     */
    public Builder addSimpleConstant(Object constant) {
      Preconditions.checkState(!immutable);
      Preconditions.checkState(
          castIfNonEmptyCollection(constant) == null,
          "Non-empty collections must be added using #addCollectionConstant: %s",
          constant);
      simpleConstants.add(constant);
      return this;
    }

    /** Convenience method, equivalent to calling {@link #addSimpleConstant} repeatedly. */
    public Builder addSimpleConstants(Object... constants) {
      for (Object constant : constants) {
        addSimpleConstant(constant);
      }
      return this;
    }

    ValueConstants build(int constantsStartTag) {
      int nextTag = constantsStartTag;
      ImmutableList.Builder<Object> constants =
          ImmutableList.builderWithExpectedSize(
              collectionConstants.size() + simpleConstants.size());
      HashMap<Class<?>, HashMap<Object, Integer>> simpleConstantsMapBuilder = new HashMap<>();
      for (Object constant : simpleConstants) {
        simpleConstantsMapBuilder
            .computeIfAbsent(constant.getClass(), k -> new HashMap<>())
            .put(constant, nextTag++);
        constants.add(constant);
      }
      HashMap<CollectionInfo, HashMap<Collection<?>, Integer>> collectionConstantsMapBuilder =
          new HashMap<>();
      for (CollectionAndCollectionInfo collectionAndCollectionInfo : collectionConstants) {
        Collection<?> collection = collectionAndCollectionInfo.getCollection();
        collectionConstantsMapBuilder
            .computeIfAbsent(collectionAndCollectionInfo.getCollectionInfo(), k -> new HashMap<>())
            .put(collection, nextTag++);
        constants.add(collection);
      }
      return new ValueConstants(
          constantsStartTag,
          toImmutableMap(simpleConstantsMapBuilder),
          toImmutableMap(collectionConstantsMapBuilder),
          constants.build());
    }
  }

  private static <K1, K2, V> ImmutableMap<K1, ImmutableMap<K2, V>> toImmutableMap(
      Map<K1, ? extends Map<K2, V>> map) {
    return map.entrySet()
        .stream()
        .collect(
            ImmutableMap.toImmutableMap(Map.Entry::getKey, e -> ImmutableMap.copyOf(e.getValue())));
  }

  @AutoValue
  abstract static class CollectionInfo {
    @SuppressWarnings("unused") // Needed for @AutoValue, want it for equality checking.
    abstract Class<? extends Collection<?>> getCollectionClass();

    abstract Class<?> getItemClass();

    abstract int getSize();

    /**
     * Gets the {@link CollectionInfo} for the given non-empty {@code collection} unless it contains
     * only {@code null} elements, in which case it returns {@code null}. This means that a
     * collection that only contains nulls cannot be a value constant.
     */
    @Nullable
    static CollectionInfo makeCollectionInfo(Collection<?> collection) {
      Preconditions.checkState(!collection.isEmpty(), collection);
      // Iterate until we find a non-null element. Shouldn't take long.
      Object elt = null;
      for (Object o : collection) {
        if (o != null) {
          elt = o;
          break;
        }
      }
      if (elt == null) {
        // Can't find the type here. Not a potential value constant.
        return null;
      }
      return new AutoValue_ValueConstants_CollectionInfo(
          castCollectionClass(collection.getClass()), elt.getClass(), collection.size());
    }
  }

  @AutoValue
  abstract static class CollectionAndCollectionInfo {
    abstract Collection<?> getCollection();

    abstract CollectionInfo getCollectionInfo();

    static CollectionAndCollectionInfo of(Collection<?> collection, CollectionInfo collectionInfo) {
      return new AutoValue_ValueConstants_CollectionAndCollectionInfo(collection, collectionInfo);
    }
  }

  @Override
  public String toString() {
    return constants.toString();
  }
}
