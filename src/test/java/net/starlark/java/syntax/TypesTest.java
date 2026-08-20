// Copyright 2025 The Bazel Authors. All Rights Reserved.
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

package net.starlark.java.syntax;

import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.util.function.Function;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests of built-in type objects. */
// TODO: #27370 - Move this to match whichever package Types.java is going to live in.
@RunWith(JUnit4.class)
public class TypesTest {
  private static final TypeContext DUMMY_CONTEXT =
      new TypeContext() {
        @Override
        public StarlarkType getStrFieldType(String name) {
          return null;
        }

        @Override
        public StarlarkType getListFieldType(String name) {
          return null;
        }

        @Override
        public StarlarkType getDictFieldType(String name) {
          return null;
        }

        @Override
        public StarlarkType getSetFieldType(String name) {
          return null;
        }

        @Override
        public StarlarkType getPredeclaredSymbolType(String name) {
          return null;
        }

        @Override
        public StarlarkType getUniversalSymbolType(String name) {
          return null;
        }
      };

  /** Asserts {@code t1} is assignable to {@code t2}. */
  private static void assertLt(StarlarkType t1, StarlarkType t2) {
    assertWithMessage("%s is expected to be assignable to %s", t1, t2)
        .that(StarlarkType.assignableFrom(t2, t1, DUMMY_CONTEXT))
        .isTrue();
  }

  /** Asserts {@code t1} is *not* assignable to {@code t2}. */
  private static void assertNotLt(StarlarkType t1, StarlarkType t2) {
    assertWithMessage("%s is expected to be *not* assignable to %s", t1, t2)
        .that(StarlarkType.assignableFrom(t2, t1, DUMMY_CONTEXT))
        .isFalse();
  }

  /** Asserts {@code t1} is assignable to {@code t2}, but not vice versa. */
  private static void assertStrictLt(StarlarkType t1, StarlarkType t2) {
    assertLt(t1, t2);
    assertNotLt(t2, t1);
  }

  /** Asserts that the given types are assignable in both directions. */
  private static void assertLtAndGt(StarlarkType... types) {
    for (int i = 0; i < types.length - 1; i++) {
      for (int j = i + 1; j < types.length; j++) {
        assertLt(types[i], types[j]);
        assertLt(types[j], types[i]);
      }
    }
  }

  /** Asserts that the given types are *not* assignable in either direction. */
  private static void assertIncomparable(StarlarkType... types) {
    for (int i = 0; i < types.length - 1; i++) {
      for (int j = i + 1; j < types.length; j++) {
        assertNotLt(types[i], types[j]);
        assertNotLt(types[j], types[i]);
      }
    }
  }

  /**
   * Asserts that the given types form a strict chain of assignability, with the ith element being
   * assignable to all jth elements where i < j, but not vice versa.
   */
  private static void assertStrictLtChain(StarlarkType... types) {
    for (int i = 0; i < types.length - 1; i++) {
      for (int j = i + 1; j < types.length; j++) {
        assertStrictLt(types[i], types[j]);
      }
    }
  }

  @Test
  public void assignability_reflexivity() {
    assertLt(Types.INT, Types.INT);
    assertLt(Types.ANY, Types.ANY);
    assertLt(Types.OBJECT, Types.OBJECT);
  }

  @Test
  public void assignability_anyPassesEitherDirection() {
    assertLtAndGt(Types.INT, Types.ANY);
  }

  @Test
  public void assignability_objectIsTop() {
    assertStrictLt(Types.INT, Types.OBJECT);
    assertLtAndGt(Types.ANY, Types.OBJECT);
  }

  @Test
  public void assignability_primitiveTypesAreIncompatible() {
    assertIncomparable(Types.INT, Types.STR);
    assertIncomparable(Types.INT, Types.FLOAT); // unlike Python
    assertIncomparable(Types.STR, Types.FLOAT);
    assertIncomparable(Types.BOOL, Types.INT); // unlike Python
    assertIncomparable(Types.NONE, Types.STR);
  }

  @Test
  public void assignability_union() {
    StarlarkType intOrStr = Types.union(Types.INT, Types.STR);
    StarlarkType intOrFloatOrStr = Types.union(Types.INT, Types.FLOAT, Types.STR);
    StarlarkType floatOrStr = Types.union(Types.FLOAT, Types.STR);
    // Assignability of a primitive type to a union
    assertLt(Types.INT, intOrStr);
    assertLt(Types.STR, intOrStr);
    assertLt(Types.ANY, intOrStr);
    assertNotLt(Types.FLOAT, intOrStr);
    assertNotLt(Types.OBJECT, intOrStr);

    // Assignability of a union to a primitive type
    assertLt(intOrStr, Types.ANY);
    assertLt(intOrStr, Types.OBJECT);
    assertNotLt(intOrStr, Types.INT);
    assertNotLt(intOrStr, Types.STR);

    // Assignability between unions
    assertLt(intOrStr, intOrStr);
    assertLt(intOrStr, intOrFloatOrStr);
    assertNotLt(intOrFloatOrStr, intOrStr);
    assertNotLt(intOrStr, floatOrStr);
    assertNotLt(floatOrStr, intOrStr);
  }

  // Application-defined Sequence subtype.
  private static class CustomSequenceType extends Types.AbstractSequenceType {
    @Override
    public StarlarkType getElementType() {
      return Types.ANY;
    }
  }

  // Application-defined Mapping subtype.
  private static class CustomMappingType extends Types.AbstractMappingType {
    @Override
    public StarlarkType getKeyType() {
      return Types.ANY;
    }

    @Override
    public StarlarkType getValueType() {
      return Types.ANY;
    }
  }

  @Test
  public void assignability_collection_subtypes() {
    StarlarkType intOrStr = Types.union(Types.INT, Types.STR);

    assertStrictLtChain(
        Types.listRvalue(Types.NEVER), // empty list literal
        Types.list(Types.INT),
        Types.sequence(Types.INT),
        Types.collection(Types.INT));

    // List rvalues are assignable to any compatible list (and supertypes)
    assertStrictLtChain(
        Types.listRvalue(Types.INT),
        Types.list(intOrStr),
        Types.sequence(intOrStr),
        Types.collection(intOrStr));
    // ... but not to incompatible collection types (e.g. mappings, dicts, sets, or
    // application-defined collection types).
    assertIncomparable(Types.listRvalue(Types.ANY), Types.mapping(Types.ANY, Types.ANY));
    assertIncomparable(Types.listRvalue(Types.ANY), Types.dict(Types.ANY, Types.ANY));
    assertIncomparable(Types.listRvalue(Types.ANY), Types.homogeneousTuple(Types.ANY));
    assertIncomparable(Types.listRvalue(Types.ANY), Types.set(Types.ANY));
    assertIncomparable(Types.listRvalue(Types.ANY), new CustomSequenceType());
    assertIncomparable(Types.listRvalue(Types.ANY), new CustomMappingType());

    assertStrictLtChain(
        Types.tuple(Types.INT, Types.STR, Types.INT, Types.STR),
        Types.homogeneousTuple(intOrStr),
        Types.sequence(intOrStr),
        Types.collection(intOrStr));

    // An empty tuple is assignable to any homogeneous tuple type.
    assertStrictLtChain(
        Types.EMPTY_TUPLE,
        Types.homogeneousTuple(Types.ANY),
        Types.sequence(Types.ANY),
        Types.collection(Types.ANY));

    assertStrictLtChain(Types.set(Types.STR), Types.collection(Types.STR));
    assertIncomparable(Types.set(Types.STR), Types.sequence(Types.STR));

    assertStrictLtChain(
        Types.dictRvalue(Types.NEVER, Types.NEVER), // empty dict literal
        Types.dict(Types.STR, Types.INT),
        Types.mapping(Types.STR, Types.INT),
        Types.collection(Types.STR));
    assertIncomparable(Types.dict(Types.STR, Types.INT), Types.sequence(Types.STR));

    // Dict rvalues are assignable to any compatible dict (and supertypes)
    assertStrictLtChain(
        Types.dictRvalue(Types.STR, Types.INT),
        Types.dict(intOrStr, intOrStr),
        Types.mapping(intOrStr, intOrStr),
        Types.collection(intOrStr));
    // ... but not to incompatible collection types (e.g. sequences, lists, sets, or
    // application-defined collection types).
    assertIncomparable(Types.dictRvalue(Types.ANY, Types.ANY), Types.sequence(Types.ANY));
    assertIncomparable(Types.dictRvalue(Types.ANY, Types.ANY), Types.list(Types.ANY));
    assertIncomparable(Types.dictRvalue(Types.ANY, Types.ANY), Types.homogeneousTuple(Types.ANY));
    assertIncomparable(Types.dictRvalue(Types.ANY, Types.ANY), Types.set(Types.ANY));
    assertIncomparable(Types.dictRvalue(Types.ANY, Types.ANY), new CustomSequenceType());
    assertIncomparable(Types.dictRvalue(Types.ANY, Types.ANY), new CustomMappingType());

    // Works with unions too.
    assertStrictLtChain(
        Types.union(Types.dict(Types.STR, Types.INT), Types.list(Types.STR)),
        Types.union(Types.collection(Types.STR), Types.collection(Types.BOOL)));
  }

  @Test
  public void assignability_homogeneousCollections_covariance() {
    // Immutable and rvalue collections: covariant in element type
    ImmutableList<Function<StarlarkType, StarlarkType>> immutableCollectionConstructors =
        ImmutableList.of(
            Types::collection, Types::sequence, Types::homogeneousTuple, Types::listRvalue);
    for (var ctor : immutableCollectionConstructors) {
      assertLtAndGt(ctor.apply(Types.INT), ctor.apply(Types.ANY));
      assertIncomparable(ctor.apply(Types.INT), ctor.apply(Types.FLOAT));
      assertStrictLtChain(
          ctor.apply(Types.NEVER),
          ctor.apply(Types.INT),
          ctor.apply(Types.NUMERIC),
          ctor.apply(Types.OBJECT));
    }

    // Mutable collections: invariant in element type.
    ImmutableList<Function<StarlarkType, StarlarkType>> mutableCollectionConstructors =
        ImmutableList.of(Types::list, Types::set);
    for (var ctor : mutableCollectionConstructors) {
      assertLtAndGt(ctor.apply(Types.INT), ctor.apply(Types.ANY));
      assertIncomparable(
          ctor.apply(Types.INT),
          ctor.apply(Types.FLOAT),
          ctor.apply(Types.NUMERIC),
          ctor.apply(Types.OBJECT));
    }
  }

  @Test
  public void assignability_fixedLengthTuple_covariance() {
    // Covariant in element types; element count must match exactly.
    assertLtAndGt(Types.tuple(Types.INT, Types.STR), Types.tuple(Types.ANY, Types.ANY));
    assertIncomparable(Types.tuple(Types.INT, Types.STR), Types.tuple(Types.ANY));
    assertIncomparable(
        Types.tuple(Types.INT, Types.STR), Types.tuple(Types.ANY, Types.ANY, Types.ANY));

    assertIncomparable(Types.tuple(Types.INT, Types.STR), Types.tuple(Types.FLOAT, Types.STR));
    assertIncomparable(Types.tuple(Types.INT, Types.STR), Types.tuple(Types.INT, Types.BOOL));
    assertIncomparable(Types.tuple(Types.INT, Types.STR), Types.tuple(Types.STR, Types.INT));
    assertIncomparable(Types.tuple(Types.INT, Types.STR), Types.tuple(Types.INT));
    assertIncomparable(
        Types.tuple(Types.INT, Types.STR), Types.tuple(Types.INT, Types.STR, Types.BOOL));

    assertStrictLtChain(
        Types.tuple(Types.INT, Types.STR),
        Types.tuple(Types.NUMERIC, Types.STR),
        Types.tuple(Types.OBJECT, Types.OBJECT));
  }

  @Test
  public void assignability_mapping_covariance() {
    // Invariant in key type (but allowing Any); covariant in value type.

    // keys
    assertLtAndGt(Types.mapping(Types.STR, Types.INT), Types.mapping(Types.ANY, Types.INT));
    assertIncomparable(
        Types.mapping(Types.STR, Types.INT),
        Types.mapping(Types.OBJECT, Types.INT),
        Types.mapping(Types.INT, Types.INT),
        Types.mapping(Types.NUMERIC, Types.INT));

    // values
    assertLtAndGt(Types.mapping(Types.STR, Types.INT), Types.mapping(Types.STR, Types.ANY));
    assertStrictLtChain(
        Types.mapping(Types.STR, Types.INT),
        Types.mapping(Types.STR, Types.NUMERIC),
        Types.mapping(Types.STR, Types.OBJECT));
  }

  @Test
  public void assignability_dict_invariance() {
    // Invariant in key and value types.

    // keys
    assertLtAndGt(Types.dict(Types.STR, Types.INT), Types.dict(Types.ANY, Types.INT));
    assertIncomparable(
        Types.dict(Types.STR, Types.INT),
        Types.dict(Types.OBJECT, Types.INT),
        Types.dict(Types.INT, Types.INT),
        Types.dict(Types.NUMERIC, Types.INT));

    // values
    assertLtAndGt(Types.dict(Types.STR, Types.INT), Types.dict(Types.STR, Types.ANY));
    assertIncomparable(
        Types.dict(Types.STR, Types.INT),
        Types.dict(Types.STR, Types.FLOAT),
        Types.dict(Types.STR, Types.NUMERIC),
        Types.dict(Types.STR, Types.OBJECT));
  }

  @Test
  public void assignability_struct() {
    assertLtAndGt(
        Types.struct(ImmutableMap.of("f", Types.INT)),
        Types.struct(ImmutableMap.of("f", Types.ANY)));
    assertLtAndGt(
        Types.partialStruct(ImmutableMap.of("f", Types.INT)),
        Types.struct(ImmutableMap.of("f", Types.INT)));

    // Order of fields is irrelevant.
    assertLtAndGt(
        Types.struct(ImmutableMap.of("f", Types.INT, "g", Types.BOOL)),
        Types.struct(ImmutableMap.of("g", Types.BOOL, "f", Types.INT)),
        Types.partialStruct(ImmutableMap.of("f", Types.INT, "g", Types.BOOL)),
        Types.partialStruct(ImmutableMap.of("g", Types.BOOL, "f", Types.INT)));

    // Partially overlapping sets of explicitly-specified fields
    assertIncomparable(
        Types.struct(ImmutableMap.of("f", Types.INT, "g", Types.INT)),
        Types.struct(ImmutableMap.of("f", Types.INT, "h", Types.INT)));
    assertLt(
        Types.partialStruct(ImmutableMap.of("f", Types.INT, "g", Types.INT)),
        Types.struct(ImmutableMap.of("f", Types.INT, "h", Types.INT)));

    // ANY_STRUCT is assignable to and from any other struct type.
    assertLtAndGt(Types.ANY_STRUCT, Types.struct(ImmutableMap.of()));
    assertLtAndGt(Types.ANY_STRUCT, Types.struct(ImmutableMap.of("f", Types.INT)));
    assertLtAndGt(Types.ANY_STRUCT, Types.partialStruct(ImmutableMap.of()));
    assertLtAndGt(Types.ANY_STRUCT, Types.partialStruct(ImmutableMap.of("f", Types.FLOAT)));

    assertLtAndGt(
        Types.partialStruct(ImmutableMap.of()),
        Types.partialStruct(ImmutableMap.of("f", Types.ANY)),
        Types.partialStruct(ImmutableMap.of("f", Types.INT)),
        Types.partialStruct(ImmutableMap.of("f", Types.INT, "g", Types.INT)),
        Types.partialStruct(ImmutableMap.of("f", Types.INT, "h", Types.INT)));
    assertIncomparable(
        Types.partialStruct(ImmutableMap.of("f", Types.INT)),
        Types.partialStruct(ImmutableMap.of("f", Types.FLOAT)));

    assertStrictLtChain(
        Types.struct(ImmutableMap.of("f", Types.INT, "g", Types.STR, "h", Types.BOOL)),
        Types.struct(ImmutableMap.of("f", Types.INT, "h", Types.ANY)),
        Types.struct(ImmutableMap.of("f", Types.union(Types.INT, Types.FLOAT))),
        Types.EMPTY_STRUCT);
  }

  @Test
  public void assignability_callable() {
    // ANY_CALLABLE is assignable to and from any other callable type.
    assertLtAndGt(
        Types.ANY_CALLABLE,
        Types.simpleCallable(ImmutableList.of(Types.INT, Types.STR), true, Types.BOOL));
    assertLtAndGt(
        Types.ANY_CALLABLE,
        Types.generalCallable(
            ImmutableList.of("x", "y"),
            ImmutableList.of(Types.INT, Types.STR),
            1,
            1,
            ImmutableSet.of("x"),
            Types.INT,
            Types.STR,
            Types.BOOL));

    // Covariant in the return type
    assertStrictLtChain(
        Types.simpleCallable(ImmutableList.of(), false, Types.INT),
        Types.generalCallable(
            ImmutableList.of(),
            ImmutableList.of(),
            0,
            0,
            ImmutableSet.of(),
            null,
            null,
            Types.NUMERIC),
        Types.simpleCallable(ImmutableList.of(), false, Types.OBJECT));

    // Contravariant in positional parameter types
    assertStrictLtChain(
        Types.simpleCallable(ImmutableList.of(Types.OBJECT, Types.OBJECT), false, Types.ANY),
        Types.generalCallable(
            ImmutableList.of("x", "y"),
            ImmutableList.of(Types.NUMERIC, Types.NUMERIC),
            2,
            2,
            ImmutableSet.of("x", "y"),
            null,
            null,
            Types.ANY),
        Types.simpleCallable(ImmutableList.of(Types.FLOAT, Types.INT), false, Types.ANY));
    assertStrictLt(
        Types.generalCallable(
            ImmutableList.of("x", "y"),
            ImmutableList.of(Types.NUMERIC, Types.NUMERIC),
            2,
            2,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("x", "y"),
            ImmutableList.of(Types.INT, Types.FLOAT),
            2,
            2,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY));

    // Contravariant in keyword parameter types
    assertStrictLt(
        Types.generalCallable(
            ImmutableList.of("kw"),
            ImmutableList.of(Types.NUMERIC),
            0,
            0,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("kw"),
            ImmutableList.of(Types.INT),
            0,
            0,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY));

    // Contravariant in varargs and kwargs; and a callable which has varargs/kwargs is assignable to
    // one that doesn't, but not vice versa.
    assertStrictLtChain(
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.STR),
            0,
            0,
            ImmutableSet.of(),
            Types.NUMERIC,
            null,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.STR),
            0,
            0,
            ImmutableSet.of(),
            Types.FLOAT,
            null,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.STR),
            0,
            0,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY));
    assertStrictLtChain(
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.STR),
            0,
            0,
            ImmutableSet.of(),
            null,
            Types.NUMERIC,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.STR),
            0,
            0,
            ImmutableSet.of(),
            null,
            Types.FLOAT,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.STR),
            0,
            0,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY));

    // Residue positional parameters fall through to varargs (which must be contravariant).
    assertStrictLt(
        Types.simpleCallable(ImmutableList.of(), true, Types.ANY),
        Types.simpleCallable(ImmutableList.of(Types.STR), false, Types.ANY));
    assertStrictLt(
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.STR),
            1,
            1,
            ImmutableSet.of(),
            Types.NUMERIC,
            null,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("x", "y"),
            ImmutableList.of(Types.STR, Types.INT),
            2,
            2,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY));

    // Residue keyword parameters fall through to kwargs (which must be contravariant).
    assertStrictLt(
        Types.generalCallable(
            ImmutableList.of("kw"),
            ImmutableList.of(Types.STR),
            1,
            1,
            ImmutableSet.of(),
            null,
            Types.NUMERIC,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("kw", "kww"),
            ImmutableList.of(Types.STR, Types.INT),
            1,
            1,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY));

    // Incompatible mandatory positionals.
    assertIncomparable(
        Types.simpleCallable(ImmutableList.of(Types.INT), false, Types.ANY),
        Types.simpleCallable(ImmutableList.of(Types.INT, Types.INT), false, Types.ANY));
    // A callable with an optional positional is assignable to one with a mandatory positional in
    // the same place, but not vice versa.
    assertStrictLt(
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.INT),
            1,
            1,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY),
        Types.simpleCallable(ImmutableList.of(Types.INT), false, Types.ANY));
    // A callable with an optional keyword is assignable to one with a mandatory keyword of the same
    // name, but not vice versa.
    assertStrictLt(
        Types.generalCallable(
            ImmutableList.of("kw"),
            ImmutableList.of(Types.INT),
            0,
            0,
            ImmutableSet.of(),
            null,
            Types.INT,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("kw"),
            ImmutableList.of(Types.INT),
            0,
            0,
            ImmutableSet.of("kw"),
            null,
            null,
            Types.ANY));

    // Names of positional-only parameters don't matter
    assertLtAndGt(
        Types.simpleCallable(ImmutableList.of(Types.INT), false, Types.ANY),
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.INT),
            1,
            1,
            ImmutableSet.of("x"),
            null,
            null,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("y"),
            ImmutableList.of(Types.INT),
            1,
            1,
            ImmutableSet.of("y"),
            null,
            null,
            Types.ANY));

    // But names of ordinary positional parameters do matter, since they may be used by keyword.
    assertStrictLt(
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.INT),
            0,
            1,
            ImmutableSet.of("x"),
            null,
            null,
            Types.ANY),
        Types.simpleCallable(ImmutableList.of(Types.INT), false, Types.ANY));
    assertIncomparable(
        Types.generalCallable(
            ImmutableList.of("x"),
            ImmutableList.of(Types.INT),
            0,
            1,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY),
        Types.generalCallable(
            ImmutableList.of("y"),
            ImmutableList.of(Types.INT),
            0,
            1,
            ImmutableSet.of(),
            null,
            null,
            Types.ANY));
  }

  @Test
  public void callable_toSignatureString() {
    // ordinary only
    assertThat(
            Types.generalCallable(
                    /* parameterNames= */ ImmutableList.of("a"),
                    /* parameterTypes= */ ImmutableList.of(Types.INT),
                    /* numPositionalOnlyParameters= */ 0,
                    /* numPositionalParameters= */ 1,
                    /* mandatoryParams= */ ImmutableSet.of("a"),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("(a: int) -> bool");
    // kwonly only
    assertThat(
            Types.generalCallable(
                    /* parameterNames= */ ImmutableList.of("a"),
                    /* parameterTypes= */ ImmutableList.of(Types.INT),
                    /* numPositionalOnlyParameters= */ 0,
                    /* numPositionalParameters= */ 0,
                    /* mandatoryParams= */ ImmutableSet.of("a"),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("(*, a: int) -> bool");
    // positional-only only
    assertThat(
            Types.generalCallable(
                    /* parameterNames= */ ImmutableList.of("x"),
                    /* parameterTypes= */ ImmutableList.of(Types.INT),
                    /* numPositionalOnlyParameters= */ 1,
                    /* numPositionalParameters= */ 1,
                    /* mandatoryParams= */ ImmutableSet.of(),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("([int], /) -> bool");
    // no params
    assertThat(
            Types.generalCallable(
                    /* parameterNames= */ ImmutableList.of(),
                    /* parameterTypes= */ ImmutableList.of(),
                    /* numPositionalOnlyParameters= */ 0,
                    /* numPositionalParameters= */ 0,
                    /* mandatoryParams= */ ImmutableSet.of(),
                    /* varargsType= */ null,
                    /* kwargsType= */ null,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo("() -> bool");
    // all kinds of params
    assertThat(
            Types.generalCallable(
                    /* parameterNames= */ ImmutableList.of("x", "a", "b", "c", "d"),
                    /* parameterTypes= */ ImmutableList.of(
                        Types.BOOL, Types.INT, Types.FLOAT, Types.NONE, Types.ANY),
                    /* numPositionalOnlyParameters= */ 1,
                    /* numPositionalParameters= */ 3,
                    /* mandatoryParams= */ ImmutableSet.of("a", "c", "x"),
                    /* varargsType= */ Types.INT,
                    /* kwargsType= */ Types.INT,
                    Types.BOOL)
                .toSignatureString())
        .isEqualTo(
            "(bool, /, a: int, b: [float], *args: int, c: None, d: [Any], **kwargs: int) -> bool");
  }
}
