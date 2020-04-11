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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.skyframe.serialization.ObjectCodec.MemoizationStrategy;
import com.google.protobuf.CodedInputStream;
import com.google.protobuf.CodedOutputStream;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.Deque;
import java.util.HashMap;
import java.util.IdentityHashMap;
import javax.annotation.Nullable;

/**
 * A framework for serializing and deserializing with memo tables. Memoization is useful both for
 * performance and, in the case of cyclic data structures, to help avoid infinite recursion.
 *
 * <p>The memo table associates each value with an integer identifier.
 *
 * <ul>
 *   <li><i>On the sending end:</i> The first time a value is to be serialized, a new id is created
 *       and a mapping for it is added to the table. The id is emitted on the wire alongside the
 *       value's serialized representation. If the same value surfaces again later on, instead of
 *       reserializing it, we just emit a backreference consisting of its id.
 *   <li><i>On the receiving end:</i> Each deserialized value is stored in the memo table along with
 *       the id it was associated with. When a backreference is read, the value in the memo table is
 *       returned instead of deserializing a new copy of that same value.
 * </ul>
 *
 * <p>Cyclic data structures can occur either naturally in complex types, or as a result of a
 * pathological Skylark program. An example of the former is how a user-defined Skylark function
 * implicitly refers to its global frame, which in turn refers to the functions defined in that
 * frame. An example of the latter is a list that the user mutates to contain itself as an element.
 * Such pathological values in Skylark are technically allowed, but they are not useful since
 * Skylark prohibits recursive function calls. They can also expose implementation bugs in code that
 * is not expecting them (b/30310522).
 *
 * <p>Ideally, to handle cyclic data structures, the serializer should add the value to the memo
 * table <em>before</em> actually performing the serialization. For example, to handle the recursive
 * list {@code L = ["A", L]}, the serializer should do the following:
 *
 * <ol>
 *   <li>add a memo mapping from {@code L} to a fresh id, {@code k}
 *   <li>emit {@code k} to the wire
 *   <li>serialize {@code L}, which means it has to
 *       <ol>
 *         <li>write its length, {@code 2}
 *         <li>serialize the string {@code "A"}
 *         <li>serialize {@code L}, but since this matches the entry added in Step 1, it just emits
 *             {@code k} as a backreference
 *       </ol>
 * </ol>
 *
 * The problem is that on the other end of the wire, the deserializer needs to associate a value
 * with the memo entry for k before {@code L} has been fully formed. To solve this, we associate k
 * with a new empty list as the initial value, then allow the deserialization logic to mutate this
 * list to form {@code L}. It is important that this is done by mutating the initial list rather
 * than by replacing it with another list, since each backreference to k creates an actual Java
 * reference to the initial object.
 *
 * <p>However, this strategy does not work for all types of values. There is no way to deserialize
 * the recursive tuple {@code T = ("A", T)}, because tuples are immutable and therefore cannot be
 * instantiated before all of their elements have been. Rather than restrict serialization to only
 * mutable types, or add a special way for deserializers to modify seemingly immutable types, we
 * simply don't memoize immutable types until after they are fully constructed. This means that
 * {@code T} is not serializable. But that's okay, because objects like {@code T} should not even be
 * able to exist (barring a hidden API or reflection). In general, all cycles must go through at
 * least one mutable type of value.
 *
 * <p>Aside from mutability, there is another potential problem: One of the types' constructors may
 * enforce an invariant that is not satisfied at the time the constructor is invoked, even though it
 * may be satisfied once construction of all objects is complete. For instance, suppose a type
 * {@code Foo} has a constructor that takes a non-empty list. Then we would fail to deserialize
 * {@code L = [Foo(L)]}, since {@code L} is initially empty. Such a list could legally be formed by
 * putting other elements in {@code L} before creating {@code Foo}, and then later removing those
 * other elements. But there's no general way for a deserializer to know how to do that. Therefore,
 * it is the caller's responsibility to ensure the following property:
 *
 * <blockquote>
 *
 * For any value that is to be serialized, if the value has children that directly or indirectly
 * contain the value, then the value must be constructible even when those children are in a
 * semi-constructed state.
 *
 * </blockquote>
 *
 * where "semi-constructed state" means any state that can be produced by the codecs for those
 * children. Other serialization systems address this issue by providing multiple hooks for types to
 * setup their invariants, but we keep the API relatively simple.
 *
 * <p>Round-tripping a value through memoized serialization and deserialization is guaranteed to
 * preserve the object graph, i.e., to not duplicate a value. For mutable types, a value can only be
 * serialized and deserialized at most once because it is memoized before recursing over its
 * children. For immutable types, although a value can be serialized multiple times, upon
 * deserialization only one copy is retained in the memo table. This is conceptually similar to how
 * Python's Pickle library <a href="https://github.com/python/cpython/blob/3.6/Lib/pickle.py#L754">
 * handles tuples</a>, although in that case they use an abstract machine whereas we do not.
 *
 * <p>Wire format, as an abstract grammar:
 *
 * <pre>{@code
 * START             -->  NoMemoContent | `NEW_VALUE` MemoContent | `BACKREF` MemoId
 * MemoContent       -->  MemoBeforeContent | MemoAfterContent
 * NoMemoContent     -->  Payload
 * MemoBeforeContent -->  MemoId Payload
 * MemoAfterContent  -->  Payload MemoId
 * MemoId            -->  int32
 * }</pre>
 *
 * where {@code Payload} is the serialized representation of the value. {@code Payload} may itself
 * contain complete memo-aware encodings of the value's children.
 */
// TODO(brandjon): Maybe make this more robust against a pathological cycle of immutable objects, so
// that instead of failing with a stack overflow, we detect the cycle and throw
// SerializationException. This requires just a little extra memo tracking for the MEMOIZE_AFTER
// case.
class Memoizer {
  private Memoizer() {}

  /** A context for serializing; wraps a memo table. Not thread-safe. */
  static class Serializer {
    private final SerializingMemoTable memo = new SerializingMemoTable();

    /**
     * Serializes an object using the given codec and current memo table state.
     *
     * @throws SerializationException on a logical error during serialization
     * @throws IOException on {@link IOException} during serialization
     */
    <T> void serialize(
        SerializationContext context,
        T obj,
        ObjectCodec<? super T> codec,
        CodedOutputStream codedOut,
        MemoizationStrategy strategy)
        throws SerializationException, IOException {
      if (strategy == MemoizationStrategy.DO_NOT_MEMOIZE) {
        codec.serialize(context, obj, codedOut);
      } else {
        // The caller already checked the table, so this is definitely a new value.
        serializeMemoContent(context, obj, codec, codedOut, strategy);
      }
    }

    @Nullable
    Integer getMemoizedIndex(Object obj) {
      return memo.lookupNullable(obj);
    }

    // Corresponds to MemoContent in the abstract grammar.
    private <T> void serializeMemoContent(
        SerializationContext context,
        T obj,
        ObjectCodec<T> codec,
        CodedOutputStream codedOut,
        MemoizationStrategy strategy)
        throws SerializationException, IOException {
      switch(strategy) {
        case MEMOIZE_BEFORE: {
          int id = memo.memoize(obj);
          codedOut.writeInt32NoTag(id);
            codec.serialize(context, obj, codedOut);
          break;
        }
        case MEMOIZE_AFTER: {
            codec.serialize(context, obj, codedOut);
          // If serializing the children caused the parent object itself to be serialized due to a
          // cycle, then there's now a memo entry for the parent. Don't overwrite it with a new id.
          Integer cylicallyCreatedId = memo.lookupNullable(obj);
          int id = (cylicallyCreatedId != null) ? cylicallyCreatedId : memo.memoize(obj);
          codedOut.writeInt32NoTag(id);
          break;
        }
        default:
          throw new AssertionError("Unreachable (strategy=" + strategy + ")");
      }
    }

    private static class SerializingMemoTable {
      private final IdentityHashMap<Object, Integer> table = new IdentityHashMap<>();

      /**
       * Adds a new value to the memo table and returns its id. The value must not already be
       * present.
       */
      private int memoize(Object value) {
        Preconditions.checkArgument(
            !table.containsKey(value),
            "Tried to memoize object '%s' multiple times", value);
        // Ids count sequentially from 0.
        int newId = table.size();
        table.put(value, newId);
        return newId;
      }

      /**
       * If the value is already memoized, return its on-the-wire id; otherwise return null. Opaque.
       *
       * <p>Beware accidental unboxing of a null result.
       */
      @Nullable
      private Integer lookupNullable(Object value) {
        return table.get(value);
      }
    }
  }

  /**
   * A context for deserializing; wraps a memo table and possibly additional data. Not thread-safe.
   */
  static class Deserializer {
    private final DeserializingMemoTable memo = new DeserializingMemoTable();
    @Nullable private Integer tagForMemoizedBefore = null;
    private final Deque<Object> memoizedBeforeStackForSanityChecking = new ArrayDeque<>();

    /**
     * Deserializes an object using the given codec and current memo table state.
     *
     * @throws SerializationException on a logical error during deserialization
     * @throws IOException on {@link IOException} during deserialization
     */
    <T> T deserialize(
        DeserializationContext context,
        ObjectCodec<? extends T> codec,
        MemoizationStrategy strategy,
        CodedInputStream codedIn)
        throws SerializationException, IOException {
      Preconditions.checkState(
          tagForMemoizedBefore == null,
          "non-null memoized-before tag %s (%s)",
          tagForMemoizedBefore,
          codec);
      if (strategy == MemoizationStrategy.DO_NOT_MEMOIZE) {
        return codec.deserialize(context, codedIn);
      } else {
          switch (strategy) {
            case MEMOIZE_BEFORE:
              return deserializeMemoBeforeContent(context, codec, codedIn);
            case MEMOIZE_AFTER:
              return deserializeMemoAfterContent(context, codec, codedIn);
            default:
              throw new AssertionError("Unreachable (strategy=" + strategy + ")");
          }
        }
    }

    Object getMemoized(int memoIndex) {
      return Preconditions.checkNotNull(memo.lookup(memoIndex), memoIndex);
    }

    private static <T> T safeCast(Object obj, ObjectCodec<T> codec) throws SerializationException {
      if (obj == null) {
        return null;
      }
      ImmutableSet<Class<? extends T>> expectedTypes =
          codec.additionalEncodedClasses().isEmpty()
              ? ImmutableSet.of(codec.getEncodedClass())
              : ImmutableSet.<Class<? extends T>>builderWithExpectedSize(
                      codec.additionalEncodedClasses().size() + 1)
                  .add(codec.getEncodedClass())
                  .addAll(codec.additionalEncodedClasses())
                  .build();
      Class<?> objectClass = obj.getClass();
      if (expectedTypes.contains(objectClass)) {
        @SuppressWarnings("unchecked")
        T checkedResult = (T) obj;
        return checkedResult;
      }
      for (Class<? extends T> expectedType : expectedTypes) {
        if (expectedType.isAssignableFrom(objectClass)) {
          return expectedType.cast(obj);
        }
      }
      throw new SerializationException(
          "Object "
              + obj
              + ") has type "
              + objectClass.getName()
              + " but expected type one of "
              + expectedTypes);
    }

    private static <T> T castedDeserialize(
        DeserializationContext context, ObjectCodec<T> codec, CodedInputStream codedIn)
        throws IOException, SerializationException {
      return safeCast(codec.deserialize(context, codedIn), codec);
    }

    <T> void registerInitialValue(T initialValue) {
      int tag =
          Preconditions.checkNotNull(
              tagForMemoizedBefore, " Not called with memoize before: %s", initialValue);
      tagForMemoizedBefore = null;
      memo.memoize(tag, initialValue);
      memoizedBeforeStackForSanityChecking.addLast(initialValue);
    }

    // Corresponds to MemoBeforeContent in the abstract grammar.
    private <T> T deserializeMemoBeforeContent(
        DeserializationContext context, ObjectCodec<T> codec, CodedInputStream codedIn)
        throws SerializationException, IOException {
      this.tagForMemoizedBefore = codedIn.readInt32();
      T value = castedDeserialize(context, codec, codedIn);
      Object initial = memoizedBeforeStackForSanityChecking.removeLast();
      if (value != initial) {
        // This indicates a bug in the particular codec subclass.
        throw new SerializationException(
            String.format(
                "codec did not return the initial instance: %s but was %s with codec %s",
                value, initial, codec));
      }
      return value;
    }

    // Corresponds to MemoAfterContent in the abstract grammar.
    private <T> T deserializeMemoAfterContent(
        DeserializationContext context, ObjectCodec<T> codec, CodedInputStream codedIn)
        throws SerializationException, IOException {
      T value = castedDeserialize(context, codec, codedIn);
      int id = codedIn.readInt32();
      // If deserializing the children caused the parent object itself to be deserialized due to
      // a cycle, then there's now a memo entry for the parent. Reuse that object, discarding
      // the one we were trying to construct here, so as to avoid creating duplicate objects in
      // the object graph.
      Object cyclicallyCreatedObject = memo.lookup(id);
      if (cyclicallyCreatedObject != null) {
        return safeCast(cyclicallyCreatedObject, codec);
      } else {
        memo.memoize(id, value);
        return value;
      }
    }

    private static class DeserializingMemoTable {

      private HashMap<Integer, Object> table = new HashMap<>();

      /**
       * Adds a new id -> object maplet to the memo table. The value must not already be present.
       */
      private void memoize(int id, Object value) {
        Preconditions.checkNotNull(value);
        Object prev = table.put(id, value);
        Preconditions.checkArgument(
            prev == null,
            "Tried to memoize id %s to object '%s', when it is already memoized to object '%s'",
            id, value, prev);
      }

      /** If the id has been memoized, returns its corresponding object. Otherwise returns null. */
      @Nullable
      private Object lookup(int id) {
        return table.get(id);
      }
    }
  }
}
