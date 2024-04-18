// Copyright 2023 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.buildjar;

import static com.google.errorprone.scanner.BuiltInCheckerSuppliers.getSuppliers;

import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableSet;
import com.google.errorprone.BugCheckerInfo;
import com.google.errorprone.bugpatterns.AlwaysThrows;
import com.google.errorprone.bugpatterns.ArrayEquals;
import com.google.errorprone.bugpatterns.ArrayFillIncompatibleType;
import com.google.errorprone.bugpatterns.ArrayHashCode;
import com.google.errorprone.bugpatterns.ArrayToString;
import com.google.errorprone.bugpatterns.ArraysAsListPrimitiveArray;
import com.google.errorprone.bugpatterns.AsyncCallableReturnsNull;
import com.google.errorprone.bugpatterns.AsyncFunctionReturnsNull;
import com.google.errorprone.bugpatterns.AutoValueBuilderDefaultsInConstructor;
import com.google.errorprone.bugpatterns.BadAnnotationImplementation;
import com.google.errorprone.bugpatterns.BadShiftAmount;
import com.google.errorprone.bugpatterns.BanJNDI;
import com.google.errorprone.bugpatterns.BoxedPrimitiveEquality;
import com.google.errorprone.bugpatterns.ChainingConstructorIgnoresParameter;
import com.google.errorprone.bugpatterns.CheckNotNullMultipleTimes;
import com.google.errorprone.bugpatterns.CheckReturnValue;
import com.google.errorprone.bugpatterns.CollectionToArraySafeParameter;
import com.google.errorprone.bugpatterns.ComparableType;
import com.google.errorprone.bugpatterns.ComparingThisWithNull;
import com.google.errorprone.bugpatterns.ComparisonOutOfRange;
import com.google.errorprone.bugpatterns.CompileTimeConstantChecker;
import com.google.errorprone.bugpatterns.ComputeIfAbsentAmbiguousReference;
import com.google.errorprone.bugpatterns.ConditionalExpressionNumericPromotion;
import com.google.errorprone.bugpatterns.ConstantOverflow;
import com.google.errorprone.bugpatterns.DangerousLiteralNullChecker;
import com.google.errorprone.bugpatterns.DeadException;
import com.google.errorprone.bugpatterns.DeadThread;
import com.google.errorprone.bugpatterns.DiscardedPostfixExpression;
import com.google.errorprone.bugpatterns.DoNotCallChecker;
import com.google.errorprone.bugpatterns.DoNotMockChecker;
import com.google.errorprone.bugpatterns.DoubleBraceInitialization;
import com.google.errorprone.bugpatterns.DuplicateMapKeys;
import com.google.errorprone.bugpatterns.EqualsHashCode;
import com.google.errorprone.bugpatterns.EqualsNaN;
import com.google.errorprone.bugpatterns.EqualsNull;
import com.google.errorprone.bugpatterns.EqualsReference;
import com.google.errorprone.bugpatterns.EqualsWrongThing;
import com.google.errorprone.bugpatterns.ForOverrideChecker;
import com.google.errorprone.bugpatterns.FunctionalInterfaceMethodChanged;
import com.google.errorprone.bugpatterns.FuturesGetCheckedIllegalExceptionType;
import com.google.errorprone.bugpatterns.FuzzyEqualsShouldNotBeUsedInEqualsMethod;
import com.google.errorprone.bugpatterns.GetClassOnAnnotation;
import com.google.errorprone.bugpatterns.GetClassOnClass;
import com.google.errorprone.bugpatterns.HashtableContains;
import com.google.errorprone.bugpatterns.IdentityBinaryExpression;
import com.google.errorprone.bugpatterns.IdentityHashMapBoxing;
import com.google.errorprone.bugpatterns.ImpossibleNullComparison;
import com.google.errorprone.bugpatterns.Incomparable;
import com.google.errorprone.bugpatterns.IncompatibleModifiersChecker;
import com.google.errorprone.bugpatterns.IndexOfChar;
import com.google.errorprone.bugpatterns.InexactVarargsConditional;
import com.google.errorprone.bugpatterns.InfiniteRecursion;
import com.google.errorprone.bugpatterns.InvalidPatternSyntax;
import com.google.errorprone.bugpatterns.InvalidTimeZoneID;
import com.google.errorprone.bugpatterns.InvalidZoneId;
import com.google.errorprone.bugpatterns.IsInstanceIncompatibleType;
import com.google.errorprone.bugpatterns.IsInstanceOfClass;
import com.google.errorprone.bugpatterns.JUnit3TestNotRun;
import com.google.errorprone.bugpatterns.JUnit4ClassAnnotationNonStatic;
import com.google.errorprone.bugpatterns.JUnit4SetUpNotRun;
import com.google.errorprone.bugpatterns.JUnit4TearDownNotRun;
import com.google.errorprone.bugpatterns.JUnit4TestNotRun;
import com.google.errorprone.bugpatterns.JUnit4TestsNotRunWithinEnclosed;
import com.google.errorprone.bugpatterns.JUnitAssertSameCheck;
import com.google.errorprone.bugpatterns.JUnitParameterMethodNotFound;
import com.google.errorprone.bugpatterns.LenientFormatStringValidation;
import com.google.errorprone.bugpatterns.LiteByteStringUtf8;
import com.google.errorprone.bugpatterns.LockOnBoxedPrimitive;
import com.google.errorprone.bugpatterns.LoopConditionChecker;
import com.google.errorprone.bugpatterns.LossyPrimitiveCompare;
import com.google.errorprone.bugpatterns.MathRoundIntLong;
import com.google.errorprone.bugpatterns.MissingSuperCall;
import com.google.errorprone.bugpatterns.MissingTestCall;
import com.google.errorprone.bugpatterns.MisusedDayOfYear;
import com.google.errorprone.bugpatterns.MisusedWeekYear;
import com.google.errorprone.bugpatterns.MixedDescriptors;
import com.google.errorprone.bugpatterns.MockitoUsage;
import com.google.errorprone.bugpatterns.ModifyingCollectionWithItself;
import com.google.errorprone.bugpatterns.MustBeClosedChecker;
import com.google.errorprone.bugpatterns.NCopiesOfChar;
import com.google.errorprone.bugpatterns.NonCanonicalStaticImport;
import com.google.errorprone.bugpatterns.NonFinalCompileTimeConstant;
import com.google.errorprone.bugpatterns.NonRuntimeAnnotation;
import com.google.errorprone.bugpatterns.NullTernary;
import com.google.errorprone.bugpatterns.NullableOnContainingClass;
import com.google.errorprone.bugpatterns.OptionalEquality;
import com.google.errorprone.bugpatterns.OptionalMapUnusedValue;
import com.google.errorprone.bugpatterns.OptionalOfRedundantMethod;
import com.google.errorprone.bugpatterns.PackageInfo;
import com.google.errorprone.bugpatterns.ParametersButNotParameterized;
import com.google.errorprone.bugpatterns.PreconditionsInvalidPlaceholder;
import com.google.errorprone.bugpatterns.PrivateSecurityContractProtoAccess;
import com.google.errorprone.bugpatterns.ProtoBuilderReturnValueIgnored;
import com.google.errorprone.bugpatterns.ProtoStringFieldReferenceEquality;
import com.google.errorprone.bugpatterns.ProtoTruthMixedDescriptors;
import com.google.errorprone.bugpatterns.ProtocolBufferOrdinal;
import com.google.errorprone.bugpatterns.RandomCast;
import com.google.errorprone.bugpatterns.RandomModInteger;
import com.google.errorprone.bugpatterns.RequiredModifiersChecker;
import com.google.errorprone.bugpatterns.RestrictedApiChecker;
import com.google.errorprone.bugpatterns.ReturnValueIgnored;
import com.google.errorprone.bugpatterns.TruthSelfEquals;
import com.google.errorprone.bugpatterns.SelfAssignment;
import com.google.errorprone.bugpatterns.SelfComparison;
import com.google.errorprone.bugpatterns.SelfEquals;
import com.google.errorprone.bugpatterns.ShouldHaveEvenArgs;
import com.google.errorprone.bugpatterns.SizeGreaterThanOrEqualsZero;
import com.google.errorprone.bugpatterns.StreamToString;
import com.google.errorprone.bugpatterns.StringBuilderInitWithChar;
import com.google.errorprone.bugpatterns.SubstringOfZero;
import com.google.errorprone.bugpatterns.SuppressWarningsDeprecated;
import com.google.errorprone.bugpatterns.TestParametersNotInitialized;
import com.google.errorprone.bugpatterns.TheoryButNoTheories;
import com.google.errorprone.bugpatterns.ThrowIfUncheckedKnownChecked;
import com.google.errorprone.bugpatterns.ThrowNull;
import com.google.errorprone.bugpatterns.TreeToString;
import com.google.errorprone.bugpatterns.TryFailThrowable;
import com.google.errorprone.bugpatterns.TypeParameterQualifier;
import com.google.errorprone.bugpatterns.UnicodeDirectionalityCharacters;
import com.google.errorprone.bugpatterns.UnicodeInCode;
import com.google.errorprone.bugpatterns.UnnecessaryTypeArgument;
import com.google.errorprone.bugpatterns.UnusedAnonymousClass;
import com.google.errorprone.bugpatterns.UnusedCollectionModifiedInPlace;
import com.google.errorprone.bugpatterns.VarTypeName;
import com.google.errorprone.bugpatterns.WrongOneof;
import com.google.errorprone.bugpatterns.XorPower;
import com.google.errorprone.bugpatterns.android.BundleDeserializationCast;
import com.google.errorprone.bugpatterns.android.IsLoggableTagLength;
import com.google.errorprone.bugpatterns.android.MislabeledAndroidString;
import com.google.errorprone.bugpatterns.android.ParcelableCreator;
import com.google.errorprone.bugpatterns.android.RectIntersectReturnValueIgnored;
import com.google.errorprone.bugpatterns.argumentselectiondefects.AutoValueConstructorOrderChecker;
import com.google.errorprone.bugpatterns.checkreturnvalue.NoCanIgnoreReturnValueOnClasses;
import com.google.errorprone.bugpatterns.collectionincompatibletype.CollectionIncompatibleType;
import com.google.errorprone.bugpatterns.collectionincompatibletype.CompatibleWithMisuse;
import com.google.errorprone.bugpatterns.collectionincompatibletype.IncompatibleArgumentType;
import com.google.errorprone.bugpatterns.flogger.FloggerFormatString;
import com.google.errorprone.bugpatterns.flogger.FloggerLogString;
import com.google.errorprone.bugpatterns.flogger.FloggerLogVarargs;
import com.google.errorprone.bugpatterns.flogger.FloggerSplitLogStatement;
import com.google.errorprone.bugpatterns.formatstring.FormatString;
import com.google.errorprone.bugpatterns.formatstring.FormatStringAnnotationChecker;
import com.google.errorprone.bugpatterns.inject.InjectOnMemberAndConstructor;
import com.google.errorprone.bugpatterns.inject.JavaxInjectOnAbstractMethod;
import com.google.errorprone.bugpatterns.inject.MisplacedScopeAnnotations;
import com.google.errorprone.bugpatterns.inject.MoreThanOneInjectableConstructor;
import com.google.errorprone.bugpatterns.inject.MoreThanOneScopeAnnotationOnClass;
import com.google.errorprone.bugpatterns.inject.OverlappingQualifierAndScopeAnnotation;
import com.google.errorprone.bugpatterns.inject.dagger.AndroidInjectionBeforeSuper;
import com.google.errorprone.bugpatterns.inject.dagger.ProvidesNull;
import com.google.errorprone.bugpatterns.inject.guice.AssistedInjectScoping;
import com.google.errorprone.bugpatterns.inject.guice.AssistedParameters;
import com.google.errorprone.bugpatterns.inject.guice.InjectOnFinalField;
import com.google.errorprone.bugpatterns.inject.guice.OverridesJavaxInjectableMethod;
import com.google.errorprone.bugpatterns.inject.guice.ProvidesMethodOutsideOfModule;
import com.google.errorprone.bugpatterns.inlineme.Validator;
import com.google.errorprone.bugpatterns.nullness.DereferenceWithNullBranch;
import com.google.errorprone.bugpatterns.nullness.NullArgumentForNonNullParameter;
import com.google.errorprone.bugpatterns.nullness.UnnecessaryCheckNotNull;
import com.google.errorprone.bugpatterns.nullness.UnsafeWildcard;
import com.google.errorprone.bugpatterns.threadsafety.GuardedByChecker;
import com.google.errorprone.bugpatterns.threadsafety.ImmutableChecker;
import com.google.errorprone.bugpatterns.time.DurationFrom;
import com.google.errorprone.bugpatterns.time.DurationGetTemporalUnit;
import com.google.errorprone.bugpatterns.time.DurationTemporalUnit;
import com.google.errorprone.bugpatterns.time.DurationToLongTimeUnit;
import com.google.errorprone.bugpatterns.time.FromTemporalAccessor;
import com.google.errorprone.bugpatterns.time.InstantTemporalUnit;
import com.google.errorprone.bugpatterns.time.InvalidJavaTimeConstant;
import com.google.errorprone.bugpatterns.time.JodaToSelf;
import com.google.errorprone.bugpatterns.time.LocalDateTemporalAmount;
import com.google.errorprone.bugpatterns.time.PeriodFrom;
import com.google.errorprone.bugpatterns.time.PeriodGetTemporalUnit;
import com.google.errorprone.bugpatterns.time.PeriodTimeMath;
import com.google.errorprone.bugpatterns.time.TemporalAccessorGetChronoField;
import com.google.errorprone.bugpatterns.time.ZoneIdOfZ;
import com.google.errorprone.scanner.BuiltInCheckerSuppliers;
import com.google.errorprone.scanner.ScannerSupplier;

/** A factory for the {@link ScannerSupplier} that supplies Error Prone checks for Bazel. */
final class BazelScannerSuppliers {
  static ScannerSupplier bazelChecks() {
    return BuiltInCheckerSuppliers.allChecks().filter(Predicates.in(ENABLED_ERRORS));
  }

  // The list of default Error Prone errors as of 2023-8-17, generated from:
  // https://github.com/google/error-prone/blob/1b1ef67c6dc59eb1060e37cf989f95312e84e76d/core/src/main/java/com/google/errorprone/scanner/BuiltInCheckerSuppliers.java#L635
  // New errors should not be enabled in this list to avoid breaking changes in java_rules release
  private static final ImmutableSet<BugCheckerInfo> ENABLED_ERRORS =
      getSuppliers(
          // keep-sorted start
          AlwaysThrows.class,
          AndroidInjectionBeforeSuper.class,
          ArrayEquals.class,
          ArrayFillIncompatibleType.class,
          ArrayHashCode.class,
          ArrayToString.class,
          ArraysAsListPrimitiveArray.class,
          AssistedInjectScoping.class,
          AssistedParameters.class,
          AsyncCallableReturnsNull.class,
          AsyncFunctionReturnsNull.class,
          AutoValueBuilderDefaultsInConstructor.class,
          AutoValueConstructorOrderChecker.class,
          BadAnnotationImplementation.class,
          BadShiftAmount.class,
          BanJNDI.class,
          BoxedPrimitiveEquality.class,
          BundleDeserializationCast.class,
          ChainingConstructorIgnoresParameter.class,
          CheckNotNullMultipleTimes.class,
          CheckReturnValue.class,
          CollectionIncompatibleType.class,
          CollectionToArraySafeParameter.class,
          ComparableType.class,
          ComparingThisWithNull.class,
          ComparisonOutOfRange.class,
          CompatibleWithMisuse.class,
          CompileTimeConstantChecker.class,
          ComputeIfAbsentAmbiguousReference.class,
          ConditionalExpressionNumericPromotion.class,
          ConstantOverflow.class,
          DangerousLiteralNullChecker.class,
          DeadException.class,
          DeadThread.class,
          DereferenceWithNullBranch.class,
          DiscardedPostfixExpression.class,
          DoNotCallChecker.class,
          DoNotMockChecker.class,
          DoubleBraceInitialization.class,
          DuplicateMapKeys.class,
          DurationFrom.class,
          DurationGetTemporalUnit.class,
          DurationTemporalUnit.class,
          DurationToLongTimeUnit.class,
          EqualsHashCode.class,
          EqualsNaN.class,
          EqualsNull.class,
          EqualsReference.class,
          EqualsWrongThing.class,
          FloggerFormatString.class,
          FloggerLogString.class,
          FloggerLogVarargs.class,
          FloggerSplitLogStatement.class,
          ForOverrideChecker.class,
          FormatString.class,
          FormatStringAnnotationChecker.class,
          FromTemporalAccessor.class,
          FunctionalInterfaceMethodChanged.class,
          FuturesGetCheckedIllegalExceptionType.class,
          FuzzyEqualsShouldNotBeUsedInEqualsMethod.class,
          GetClassOnAnnotation.class,
          GetClassOnClass.class,
          GuardedByChecker.class,
          HashtableContains.class,
          IdentityBinaryExpression.class,
          IdentityHashMapBoxing.class,
          ImmutableChecker.class,
          ImpossibleNullComparison.class,
          Incomparable.class,
          IncompatibleArgumentType.class,
          IncompatibleModifiersChecker.class,
          IndexOfChar.class,
          InexactVarargsConditional.class,
          InfiniteRecursion.class,
          InjectOnFinalField.class,
          InjectOnMemberAndConstructor.class,
          InstantTemporalUnit.class,
          InvalidJavaTimeConstant.class,
          InvalidPatternSyntax.class,
          InvalidTimeZoneID.class,
          InvalidZoneId.class,
          IsInstanceIncompatibleType.class,
          IsInstanceOfClass.class,
          IsLoggableTagLength.class,
          JUnit3TestNotRun.class,
          JUnit4ClassAnnotationNonStatic.class,
          JUnit4SetUpNotRun.class,
          JUnit4TearDownNotRun.class,
          JUnit4TestNotRun.class,
          JUnit4TestsNotRunWithinEnclosed.class,
          JUnitAssertSameCheck.class,
          JUnitParameterMethodNotFound.class,
          JavaxInjectOnAbstractMethod.class,
          JodaToSelf.class,
          LenientFormatStringValidation.class,
          LiteByteStringUtf8.class,
          LocalDateTemporalAmount.class,
          LockOnBoxedPrimitive.class,
          LoopConditionChecker.class,
          LossyPrimitiveCompare.class,
          MathRoundIntLong.class,
          MislabeledAndroidString.class,
          MisplacedScopeAnnotations.class,
          MissingSuperCall.class,
          MissingTestCall.class,
          MisusedDayOfYear.class,
          MisusedWeekYear.class,
          MixedDescriptors.class,
          MockitoUsage.class,
          ModifyingCollectionWithItself.class,
          MoreThanOneInjectableConstructor.class,
          MoreThanOneScopeAnnotationOnClass.class,
          MustBeClosedChecker.class,
          NCopiesOfChar.class,
          NoCanIgnoreReturnValueOnClasses.class,
          NonCanonicalStaticImport.class,
          NonFinalCompileTimeConstant.class,
          NonRuntimeAnnotation.class,
          NullArgumentForNonNullParameter.class,
          NullTernary.class,
          NullableOnContainingClass.class,
          OptionalEquality.class,
          OptionalMapUnusedValue.class,
          OptionalOfRedundantMethod.class,
          OverlappingQualifierAndScopeAnnotation.class,
          OverridesJavaxInjectableMethod.class,
          PackageInfo.class,
          ParametersButNotParameterized.class,
          ParcelableCreator.class,
          PeriodFrom.class,
          PeriodGetTemporalUnit.class,
          PeriodTimeMath.class,
          PreconditionsInvalidPlaceholder.class,
          PrivateSecurityContractProtoAccess.class,
          ProtoBuilderReturnValueIgnored.class,
          ProtoStringFieldReferenceEquality.class,
          ProtoTruthMixedDescriptors.class,
          ProtocolBufferOrdinal.class,
          ProvidesMethodOutsideOfModule.class,
          ProvidesNull.class,
          RandomCast.class,
          RandomModInteger.class,
          RectIntersectReturnValueIgnored.class,
          RequiredModifiersChecker.class,
          RestrictedApiChecker.class,
          ReturnValueIgnored.class,
          // If you got a build error here, remove the rewrite in
          // devtools/blaze/bazel/admin/copybara/copy.bara.sky.
          TruthSelfEquals.class,
          SelfAssignment.class,
          SelfComparison.class,
          SelfEquals.class,
          ShouldHaveEvenArgs.class,
          SizeGreaterThanOrEqualsZero.class,
          StreamToString.class,
          StringBuilderInitWithChar.class,
          SubstringOfZero.class,
          SuppressWarningsDeprecated.class,
          TemporalAccessorGetChronoField.class,
          TestParametersNotInitialized.class,
          TheoryButNoTheories.class,
          ThrowIfUncheckedKnownChecked.class,
          ThrowNull.class,
          TreeToString.class,
          TryFailThrowable.class,
          TypeParameterQualifier.class,
          UnicodeDirectionalityCharacters.class,
          UnicodeInCode.class,
          UnnecessaryCheckNotNull.class,
          UnnecessaryTypeArgument.class,
          UnsafeWildcard.class,
          UnusedAnonymousClass.class,
          UnusedCollectionModifiedInPlace.class,
          Validator.class,
          VarTypeName.class,
          WrongOneof.class,
          XorPower.class,
          ZoneIdOfZ.class
          // keep-sorted end
          );

  private BazelScannerSuppliers() {}
}
