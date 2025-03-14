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
import com.google.errorprone.bugpatterns.*;
import com.google.errorprone.bugpatterns.SelfAssertion;
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
          SelfAssertion.class,
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
