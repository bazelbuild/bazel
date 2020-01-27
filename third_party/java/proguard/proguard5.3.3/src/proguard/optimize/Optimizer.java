/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2017 Eric Lafortune @ GuardSquare
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.optimize;

import proguard.*;
import proguard.classfile.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.visitor.*;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.visitor.*;
import proguard.classfile.util.MethodLinker;
import proguard.classfile.visitor.*;
import proguard.evaluation.*;
import proguard.evaluation.value.*;
import proguard.optimize.evaluation.*;
import proguard.optimize.info.*;
import proguard.optimize.peephole.*;
import proguard.util.*;

import java.io.IOException;
import java.util.*;

/**
 * This class optimizes class pools according to a given configuration.
 *
 * @author Eric Lafortune
 */
public class Optimizer
{
    private static final String CLASS_MARKING_FINAL            = "class/marking/final";
    private static final String CLASS_UNBOXING_ENUM            = "class/unboxing/enum";
    private static final String CLASS_MERGING_VERTICAL         = "class/merging/vertical";
    private static final String CLASS_MERGING_HORIZONTAL       = "class/merging/horizontal";
    private static final String FIELD_REMOVAL_WRITEONLY        = "field/removal/writeonly";
    private static final String FIELD_MARKING_PRIVATE          = "field/marking/private";
    private static final String FIELD_PROPAGATION_VALUE        = "field/propagation/value";
    private static final String METHOD_MARKING_PRIVATE         = "method/marking/private";
    private static final String METHOD_MARKING_STATIC          = "method/marking/static";
    private static final String METHOD_MARKING_FINAL           = "method/marking/final";
    private static final String METHOD_REMOVAL_PARAMETER       = "method/removal/parameter";
    private static final String METHOD_PROPAGATION_PARAMETER   = "method/propagation/parameter";
    private static final String METHOD_PROPAGATION_RETURNVALUE = "method/propagation/returnvalue";
    private static final String METHOD_INLINING_SHORT          = "method/inlining/short";
    private static final String METHOD_INLINING_UNIQUE         = "method/inlining/unique";
    private static final String METHOD_INLINING_TAILRECURSION  = "method/inlining/tailrecursion";
    private static final String CODE_MERGING                   = "code/merging";
    private static final String CODE_SIMPLIFICATION_VARIABLE   = "code/simplification/variable";
    private static final String CODE_SIMPLIFICATION_ARITHMETIC = "code/simplification/arithmetic";
    private static final String CODE_SIMPLIFICATION_CAST       = "code/simplification/cast";
    private static final String CODE_SIMPLIFICATION_FIELD      = "code/simplification/field";
    private static final String CODE_SIMPLIFICATION_BRANCH     = "code/simplification/branch";
    private static final String CODE_SIMPLIFICATION_STRING     = "code/simplification/string";
    private static final String CODE_SIMPLIFICATION_ADVANCED   = "code/simplification/advanced";
    private static final String CODE_REMOVAL_ADVANCED          = "code/removal/advanced";
    private static final String CODE_REMOVAL_SIMPLE            = "code/removal/simple";
    private static final String CODE_REMOVAL_VARIABLE          = "code/removal/variable";
    private static final String CODE_REMOVAL_EXCEPTION         = "code/removal/exception";
    private static final String CODE_ALLOCATION_VARIABLE       = "code/allocation/variable";


    public static final String[] OPTIMIZATION_NAMES = new String[]
    {
        CLASS_MARKING_FINAL,
        CLASS_MERGING_VERTICAL,
        CLASS_MERGING_HORIZONTAL,
        FIELD_REMOVAL_WRITEONLY,
        FIELD_MARKING_PRIVATE,
        FIELD_PROPAGATION_VALUE,
        METHOD_MARKING_PRIVATE,
        METHOD_MARKING_STATIC,
        METHOD_MARKING_FINAL,
        METHOD_REMOVAL_PARAMETER,
        METHOD_PROPAGATION_PARAMETER,
        METHOD_PROPAGATION_RETURNVALUE,
        METHOD_INLINING_SHORT,
        METHOD_INLINING_UNIQUE,
        METHOD_INLINING_TAILRECURSION,
        CODE_MERGING,
        CODE_SIMPLIFICATION_VARIABLE,
        CODE_SIMPLIFICATION_ARITHMETIC,
        CODE_SIMPLIFICATION_CAST,
        CODE_SIMPLIFICATION_FIELD,
        CODE_SIMPLIFICATION_BRANCH,
        CODE_SIMPLIFICATION_STRING,
        CODE_SIMPLIFICATION_ADVANCED,
        CODE_REMOVAL_ADVANCED,
        CODE_REMOVAL_SIMPLE,
        CODE_REMOVAL_VARIABLE,
        CODE_REMOVAL_EXCEPTION,
        CODE_ALLOCATION_VARIABLE,
    };


    private final Configuration configuration;


    /**
     * Creates a new Optimizer.
     */
    public Optimizer(Configuration configuration)
    {
        this.configuration = configuration;
    }


    /**
     * Performs optimization of the given program class pool.
     */
    public boolean execute(ClassPool programClassPool,
                           ClassPool libraryClassPool) throws IOException
    {
        // Check if we have at least some keep commands.
        if (configuration.keep         == null &&
            configuration.applyMapping == null &&
            configuration.printMapping == null)
        {
            throw new IOException("You have to specify '-keep' options for the optimization step.");
        }

        // Create a matcher for filtering optimizations.
        StringMatcher filter = configuration.optimizations != null ?
            new ListParser(new NameParser()).parse(configuration.optimizations) :
            new ConstantMatcher(true);

        boolean classMarkingFinal            = filter.matches(CLASS_MARKING_FINAL);
        boolean classUnboxingEnum            = filter.matches(CLASS_UNBOXING_ENUM);
        boolean classMergingVertical         = filter.matches(CLASS_MERGING_VERTICAL);
        boolean classMergingHorizontal       = filter.matches(CLASS_MERGING_HORIZONTAL);
        boolean fieldRemovalWriteonly        = filter.matches(FIELD_REMOVAL_WRITEONLY);
        boolean fieldMarkingPrivate          = filter.matches(FIELD_MARKING_PRIVATE);
        boolean fieldPropagationValue        = filter.matches(FIELD_PROPAGATION_VALUE);
        boolean methodMarkingPrivate         = filter.matches(METHOD_MARKING_PRIVATE);
        boolean methodMarkingStatic          = filter.matches(METHOD_MARKING_STATIC);
        boolean methodMarkingFinal           = filter.matches(METHOD_MARKING_FINAL);
        boolean methodRemovalParameter       = filter.matches(METHOD_REMOVAL_PARAMETER);
        boolean methodPropagationParameter   = filter.matches(METHOD_PROPAGATION_PARAMETER);
        boolean methodPropagationReturnvalue = filter.matches(METHOD_PROPAGATION_RETURNVALUE);
        boolean methodInliningShort          = filter.matches(METHOD_INLINING_SHORT);
        boolean methodInliningUnique         = filter.matches(METHOD_INLINING_UNIQUE);
        boolean methodInliningTailrecursion  = filter.matches(METHOD_INLINING_TAILRECURSION);
        boolean codeMerging                  = filter.matches(CODE_MERGING);
        boolean codeSimplificationVariable   = filter.matches(CODE_SIMPLIFICATION_VARIABLE);
        boolean codeSimplificationArithmetic = filter.matches(CODE_SIMPLIFICATION_ARITHMETIC);
        boolean codeSimplificationCast       = filter.matches(CODE_SIMPLIFICATION_CAST);
        boolean codeSimplificationField      = filter.matches(CODE_SIMPLIFICATION_FIELD);
        boolean codeSimplificationBranch     = filter.matches(CODE_SIMPLIFICATION_BRANCH);
        boolean codeSimplificationString     = filter.matches(CODE_SIMPLIFICATION_STRING);
        boolean codeSimplificationAdvanced   = filter.matches(CODE_SIMPLIFICATION_ADVANCED);
        boolean codeRemovalAdvanced          = filter.matches(CODE_REMOVAL_ADVANCED);
        boolean codeRemovalSimple            = filter.matches(CODE_REMOVAL_SIMPLE);
        boolean codeRemovalVariable          = filter.matches(CODE_REMOVAL_VARIABLE);
        boolean codeRemovalException         = filter.matches(CODE_REMOVAL_EXCEPTION);
        boolean codeAllocationVariable       = filter.matches(CODE_ALLOCATION_VARIABLE);

        // Create counters to count the numbers of optimizations.
        ClassCounter       classMarkingFinalCounter            = new ClassCounter();
        ClassCounter       classUnboxingEnumCounter            = new ClassCounter();
        ClassCounter       classMergingVerticalCounter         = new ClassCounter();
        ClassCounter       classMergingHorizontalCounter       = new ClassCounter();
        MemberCounter      fieldRemovalWriteonlyCounter        = new MemberCounter();
        MemberCounter      fieldMarkingPrivateCounter          = new MemberCounter();
        MemberCounter      fieldPropagationValueCounter        = new MemberCounter();
        MemberCounter      methodMarkingPrivateCounter         = new MemberCounter();
        MemberCounter      methodMarkingStaticCounter          = new MemberCounter();
        MemberCounter      methodMarkingFinalCounter           = new MemberCounter();
        MemberCounter      methodRemovalParameterCounter       = new MemberCounter();
        MemberCounter      methodPropagationParameterCounter   = new MemberCounter();
        MemberCounter      methodPropagationReturnvalueCounter = new MemberCounter();
        InstructionCounter methodInliningShortCounter          = new InstructionCounter();
        InstructionCounter methodInliningUniqueCounter         = new InstructionCounter();
        InstructionCounter methodInliningTailrecursionCounter  = new InstructionCounter();
        InstructionCounter codeMergingCounter                  = new InstructionCounter();
        InstructionCounter codeSimplificationVariableCounter   = new InstructionCounter();
        InstructionCounter codeSimplificationArithmeticCounter = new InstructionCounter();
        InstructionCounter codeSimplificationCastCounter       = new InstructionCounter();
        InstructionCounter codeSimplificationFieldCounter      = new InstructionCounter();
        InstructionCounter codeSimplificationBranchCounter     = new InstructionCounter();
        InstructionCounter codeSimplificationStringCounter     = new InstructionCounter();
        InstructionCounter codeSimplificationAdvancedCounter   = new InstructionCounter();
        InstructionCounter deletedCounter                      = new InstructionCounter();
        InstructionCounter addedCounter                        = new InstructionCounter();
        MemberCounter      codeRemovalVariableCounter          = new MemberCounter();
        ExceptionCounter   codeRemovalExceptionCounter         = new ExceptionCounter();
        MemberCounter      codeAllocationVariableCounter       = new MemberCounter();
        MemberCounter      initializerFixCounter1              = new MemberCounter();
        MemberCounter      initializerFixCounter2              = new MemberCounter();

        // Some optimizations are required by other optimizations.
        codeSimplificationAdvanced =
            codeSimplificationAdvanced ||
            fieldPropagationValue      ||
            methodPropagationParameter ||
            methodPropagationReturnvalue;

        codeRemovalAdvanced =
            codeRemovalAdvanced   ||
            fieldRemovalWriteonly ||
            methodMarkingStatic   ||
            methodRemovalParameter;

        codeRemovalSimple =
            codeRemovalSimple ||
            codeSimplificationBranch;

        codeRemovalException =
            codeRemovalException ||
            codeRemovalAdvanced ||
            codeRemovalSimple;

        // Clean up any old visitor info.
        programClassPool.classesAccept(new ClassCleaner());
        libraryClassPool.classesAccept(new ClassCleaner());

        // Link all methods that should get the same optimization info.
        programClassPool.classesAccept(new BottomClassFilter(
                                       new MethodLinker()));
        libraryClassPool.classesAccept(new BottomClassFilter(
                                       new MethodLinker()));

        // Create a visitor for marking the seeds.
        KeepMarker keepMarker = new KeepMarker();
        ClassPoolVisitor classPoolvisitor =
            ClassSpecificationVisitorFactory.createClassPoolVisitor(configuration.keep,
                                                                    keepMarker,
                                                                    keepMarker,
                                                                    false,
                                                                    true,
                                                                    false);
        // Mark the seeds.
        programClassPool.accept(classPoolvisitor);
        libraryClassPool.accept(classPoolvisitor);

        // All library classes and library class members remain unchanged.
        libraryClassPool.classesAccept(keepMarker);
        libraryClassPool.classesAccept(new AllMemberVisitor(keepMarker));

        // We also keep all classes that are involved in .class constructs.
        // We're not looking at enum classes though, so they can be simplified.
        programClassPool.classesAccept(
            new ClassAccessFilter(0, ClassConstants.ACC_ENUM,
            new AllMethodVisitor(
            new AllAttributeVisitor(
            new AllInstructionVisitor(
            new DotClassClassVisitor(keepMarker))))));

        // We also keep all classes that are accessed dynamically.
        programClassPool.classesAccept(
            new AllConstantVisitor(
            new ConstantTagFilter(ClassConstants.CONSTANT_String,
            new ReferencedClassVisitor(keepMarker))));

        // We also keep all class members that are accessed dynamically.
        programClassPool.classesAccept(
            new AllConstantVisitor(
            new ConstantTagFilter(ClassConstants.CONSTANT_String,
            new ReferencedMemberVisitor(keepMarker))));

        // We also keep all bootstrap method signatures.
        programClassPool.classesAccept(
            new ClassVersionFilter(ClassConstants.CLASS_VERSION_1_7,
            new AllAttributeVisitor(
            new AttributeNameFilter(ClassConstants.ATTR_BootstrapMethods,
            new AllBootstrapMethodInfoVisitor(
            new BootstrapMethodHandleTraveler(
            new MethodrefTraveler(
            new ReferencedMemberVisitor(keepMarker))))))));

        // We also keep all bootstrap method arguments that point to methods.
        // These arguments are typically the method handles for
        // java.lang.invoke.LambdaMetafactory#metafactory, which provides the
        // implementations for closures.
        programClassPool.classesAccept(
            new ClassVersionFilter(ClassConstants.CLASS_VERSION_1_7,
            new AllAttributeVisitor(
            new AttributeNameFilter(ClassConstants.ATTR_BootstrapMethods,
            new AllBootstrapMethodInfoVisitor(
            new BootstrapMethodArgumentVisitor(
            new MethodrefTraveler(
            new ReferencedMemberVisitor(keepMarker))))))));

        // We also keep all classes (and their methods) returned by dynamic
        // method invocations. They may return dynamic implementations of
        // interfaces that otherwise appear unused.
        programClassPool.classesAccept(
            new ClassVersionFilter(ClassConstants.CLASS_VERSION_1_7,
            new AllConstantVisitor(
            new DynamicReturnedClassVisitor(
            new MultiClassVisitor(new ClassVisitor[]
            {
                keepMarker,
                new AllMemberVisitor(keepMarker)
            })))));

        // Attach some optimization info to all classes and class members, so
        // it can be filled out later.
        programClassPool.classesAccept(new ClassOptimizationInfoSetter());

        programClassPool.classesAccept(new AllMemberVisitor(
                                       new MemberOptimizationInfoSetter()));

        if (configuration.assumeNoSideEffects != null)
        {
            // Create a visitor for marking methods that don't have any side effects.
            NoSideEffectMethodMarker noSideEffectMethodMarker = new NoSideEffectMethodMarker();
            ClassPoolVisitor noClassPoolvisitor =
                ClassSpecificationVisitorFactory.createClassPoolVisitor(configuration.assumeNoSideEffects,
                                                                        null,
                                                                        noSideEffectMethodMarker);

            // Mark the seeds.
            programClassPool.accept(noClassPoolvisitor);
            libraryClassPool.accept(noClassPoolvisitor);
        }

        if (classMarkingFinal)
        {
            // Make classes final, whereever possible.
            programClassPool.classesAccept(
                new ClassFinalizer(classMarkingFinalCounter));
        }

        if (methodMarkingFinal)
        {
            // Make methods final, whereever possible.
            programClassPool.classesAccept(
                new ClassAccessFilter(0, ClassConstants.ACC_INTERFACE,
                new AllMethodVisitor(
                new MethodFinalizer(methodMarkingFinalCounter))));
        }

        if (fieldRemovalWriteonly)
        {
            // Mark all fields that are write-only.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new AllInstructionVisitor(
                new ReadWriteFieldMarker()))));

            // Count the write-only fields.
            programClassPool.classesAccept(
                new AllFieldVisitor(
                new WriteOnlyFieldFilter(fieldRemovalWriteonlyCounter)));
        }
        else
        {
            // Mark all fields as read/write.
            programClassPool.classesAccept(
                new AllFieldVisitor(
                new ReadWriteFieldMarker()));
        }

        if (classUnboxingEnum)
        {
             ClassCounter counter = new ClassCounter();

            // Mark all final enums that qualify as simple enums.
            programClassPool.classesAccept(
                new ClassAccessFilter(ClassConstants.ACC_FINAL |
                                      ClassConstants.ACC_ENUM, 0,
                new OptimizationInfoClassFilter(
                new SimpleEnumClassChecker())));

            // Count the preliminary number of simple enums.
            programClassPool.classesAccept(
                new SimpleEnumFilter(counter));

            // Only continue checking simple enums if there are any candidates.
            if (counter.getCount() > 0)
            {
                // Unmark all simple enums that are explicitly used as objects.
                programClassPool.classesAccept(
                    new SimpleEnumUseChecker());

                // Unmark all simple enums that are used in descriptors of
                // kept class members. Changing their names could upset
                // the name parameters of invokedynamic instructions.
                programClassPool.classesAccept(
                    new SimpleEnumFilter(null,
                    new AllMemberVisitor(
                    new KeptMemberFilter(
                    new MemberDescriptorReferencedClassVisitor(
                    new OptimizationInfoClassFilter(
                    new SimpleEnumMarker(false)))))));

                // Count the definitive number of simple enums.
                programClassPool.classesAccept(
                    new SimpleEnumFilter(classUnboxingEnumCounter));

                // Only start handling simple enums if there are any.
                if (classUnboxingEnumCounter.getCount() > 0)
                {
                    // Simplify the use of the enum classes in code.
                    programClassPool.classesAccept(
                        new AllMethodVisitor(
                        new AllAttributeVisitor(
                        new SimpleEnumUseSimplifier())));

                    // Simplify the static initializers of simple enum classes.
                    programClassPool.classesAccept(
                        new SimpleEnumFilter(
                        new SimpleEnumClassSimplifier()));

                    // Simplify the use of the enum classes in descriptors.
                    programClassPool.classesAccept(
                        new SimpleEnumDescriptorSimplifier());

                    // Update references to class members with simple enum classes.
                    programClassPool.classesAccept(new MemberReferenceFixer());
                }
            }
        }

        // Mark all used parameters, including the 'this' parameters.
        programClassPool.classesAccept(
            new AllMethodVisitor(
            new OptimizationInfoMemberFilter(
            new ParameterUsageMarker(!methodMarkingStatic,
                                     !methodRemovalParameter))));

        // Mark all classes that have static initializers.
        programClassPool.classesAccept(new StaticInitializerContainingClassMarker());

        // Mark all methods that have side effects.
        programClassPool.accept(new SideEffectMethodMarker());

//        System.out.println("Optimizer.execute: before evaluation simplification");
//        programClassPool.classAccept("abc/Def", new NamedMethodVisitor("abc", null, new ClassPrinter()));

        // Perform partial evaluation for filling out fields, method parameters,
        // and method return values, so they can be propagated.
        if (fieldPropagationValue      ||
            methodPropagationParameter ||
            methodPropagationReturnvalue)
        {
            // We'll create values to be stored with fields, method parameters,
            // and return values.
            ValueFactory valueFactory         = new ParticularValueFactory();
            ValueFactory detailedValueFactory = new DetailedValueFactory();

            InvocationUnit storingInvocationUnit =
                new StoringInvocationUnit(valueFactory,
                                          fieldPropagationValue,
                                          methodPropagationParameter,
                                          methodPropagationReturnvalue);

            // Evaluate synthetic classes in more detail, notably to propagate
            // the arrays of the classes generated for enum switch statements.
            programClassPool.classesAccept(
                new ClassAccessFilter(ClassConstants.ACC_SYNTHETIC, 0,
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new PartialEvaluator(detailedValueFactory, storingInvocationUnit, false)))));

            // Evaluate non-synthetic classes.
            programClassPool.classesAccept(
                new ClassAccessFilter(0, ClassConstants.ACC_SYNTHETIC,
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new PartialEvaluator(valueFactory, storingInvocationUnit, false)))));

            if (fieldPropagationValue)
            {
                // Count the constant fields.
                programClassPool.classesAccept(
                    new AllFieldVisitor(
                    new ConstantMemberFilter(fieldPropagationValueCounter)));
            }

            if (methodPropagationParameter)
            {
                // Count the constant method parameters.
                programClassPool.classesAccept(
                    new AllMethodVisitor(
                    new ConstantParameterFilter(methodPropagationParameterCounter)));
            }

            if (methodPropagationReturnvalue)
            {
                // Count the constant method return values.
                programClassPool.classesAccept(
                    new AllMethodVisitor(
                    new ConstantMemberFilter(methodPropagationReturnvalueCounter)));
            }

            if (classUnboxingEnumCounter.getCount() > 0)
            {
                // Propagate the simple enum constant counts.
                programClassPool.classesAccept(
                    new SimpleEnumFilter(
                    new SimpleEnumArrayPropagator()));
            }

            if (codeSimplificationAdvanced)
            {
                // Fill out constants into the arrays of synthetic classes,
                // notably the arrays of the classes generated for enum switch
                // statements.
                InvocationUnit loadingInvocationUnit =
                    new LoadingInvocationUnit(valueFactory,
                                              fieldPropagationValue,
                                              methodPropagationParameter,
                                              methodPropagationReturnvalue);

                programClassPool.classesAccept(
                    new ClassAccessFilter(ClassConstants.ACC_SYNTHETIC, 0,
                    new AllMethodVisitor(
                    new AllAttributeVisitor(
                    new PartialEvaluator(valueFactory, loadingInvocationUnit, false)))));
            }
        }

        // Perform partial evaluation again, now loading any previously stored
        // values for fields, method parameters, and method return values.
        ValueFactory valueFactory = new IdentifiedValueFactory();

        InvocationUnit loadingInvocationUnit =
            new LoadingInvocationUnit(valueFactory,
                                      fieldPropagationValue,
                                      methodPropagationParameter,
                                      methodPropagationReturnvalue);

        if (codeSimplificationAdvanced)
        {
            // Simplify based on partial evaluation, propagating constant
            // field values, method parameter values, and return values.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new EvaluationSimplifier(
                new PartialEvaluator(valueFactory, loadingInvocationUnit, false),
                codeSimplificationAdvancedCounter))));
        }

        if (codeRemovalAdvanced)
        {
            // Remove code based on partial evaluation, also removing unused
            // parameters from method invocations, and making methods static
            // if possible.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new EvaluationShrinker(
                new PartialEvaluator(valueFactory, loadingInvocationUnit, !codeSimplificationAdvanced),
                deletedCounter, addedCounter))));
        }

        if (methodRemovalParameter)
        {
            // Shrink the parameters in the method descriptors.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new OptimizationInfoMemberFilter(
                new MethodDescriptorShrinker())));
        }

        if (methodMarkingStatic)
        {
            // Make all non-static methods that don't require the 'this'
            // parameter static.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new OptimizationInfoMemberFilter(
                new MemberAccessFilter(0, ClassConstants.ACC_STATIC,
                new MethodStaticizer(methodMarkingStaticCounter)))));
        }

        if (methodRemovalParameter)
        {
            // Fix all references to class members.
            // This operation also updates the stack sizes.
            programClassPool.classesAccept(
                new MemberReferenceFixer());

            // Remove unused bootstrap method arguments.
            programClassPool.classesAccept(
                new AllAttributeVisitor(
                new AllBootstrapMethodInfoVisitor(
                new BootstrapMethodArgumentShrinker())));
        }

        if (methodRemovalParameter ||
            methodMarkingPrivate   ||
            methodMarkingStatic)
        {
            // Remove all unused parameters from the byte code, shifting all
            // remaining variables.
            // This operation also updates the local variable frame sizes.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new ParameterShrinker(methodRemovalParameterCounter))));
        }
        else if (codeRemovalAdvanced)
        {
            // Just update the local variable frame sizes.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new StackSizeUpdater())));
        }

        if (methodRemovalParameter &&
            methodRemovalParameterCounter.getCount() > 0)
        {
            // Tweak the descriptors of duplicate initializers, due to removed
            // method parameters.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new DuplicateInitializerFixer(initializerFixCounter1)));

            if (initializerFixCounter1.getCount() > 0)
            {
                // Fix all invocations of tweaked initializers.
                programClassPool.classesAccept(
                    new AllMethodVisitor(
                    new AllAttributeVisitor(
                    new DuplicateInitializerInvocationFixer(addedCounter))));

                // Fix all references to tweaked initializers.
                programClassPool.classesAccept(new MemberReferenceFixer());
            }
        }

        //// Specializing the class member descriptors seems to increase the
        //// class file size, on average.
        //// Specialize all class member descriptors.
        //programClassPool.classesAccept(new AllMemberVisitor(
        //                               new OptimizationInfoMemberFilter(
        //                               new MemberDescriptorSpecializer())));
        //
        //// Fix all references to classes, for MemberDescriptorSpecializer.
        //programClassPool.classesAccept(new AllMemberVisitor(
        //                               new OptimizationInfoMemberFilter(
        //                               new ClassReferenceFixer(true))));

        // Mark all classes with package visible members.
        // Mark all exception catches of methods.
        // Count all method invocations.
        // Mark super invocations and other access of methods.
        StackSizeComputer stackSizeComputer = new StackSizeComputer();

        programClassPool.classesAccept(
            new MultiClassVisitor(
            new ClassVisitor[]
            {
                // Mark classes.
                new PackageVisibleMemberContainingClassMarker(),
                new AllConstantVisitor(
                new PackageVisibleMemberInvokingClassMarker()),

                // Mark methods.
                new AllMethodVisitor(
                new OptimizationInfoMemberFilter(
                new AllAttributeVisitor(
                new MultiAttributeVisitor(
                new AttributeVisitor[]
                {
                    stackSizeComputer,
                    new CatchExceptionMarker(),
                    new AllInstructionVisitor(
                    new MultiInstructionVisitor(
                    new InstructionVisitor[]
                    {
                        new SuperInvocationMarker(),
                        new DynamicInvocationMarker(),
                        new BackwardBranchMarker(),
                        new AccessMethodMarker(),
                        new NonEmptyStackReturnMarker(stackSizeComputer),
                    })),
                })))),

                // Mark referenced classes and methods.
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new MultiAttributeVisitor(
                new AttributeVisitor[]
                {
                    new AllExceptionInfoVisitor(
                    new ExceptionHandlerConstantVisitor(
                    new ReferencedClassVisitor(
                    new CaughtClassMarker()))),

                    new AllInstructionVisitor(
                    new MultiInstructionVisitor(
                    new InstructionVisitor[]
                    {
                        new InstantiationClassMarker(),
                        new InstanceofClassMarker(),
                        new DotClassMarker(),
                        new MethodInvocationMarker()
                    })),
                })))
            }));

        if (classMergingVertical)
        {
            // Merge subclasses up into their superclasses or
            // merge interfaces down into their implementing classes.
            programClassPool.classesAccept(
                new VerticalClassMerger(configuration.allowAccessModification,
                                        configuration.mergeInterfacesAggressively,
                                        classMergingVerticalCounter));
        }

        if (classMergingHorizontal)
        {
            // Merge classes into their sibling classes.
            programClassPool.classesAccept(
                new HorizontalClassMerger(configuration.allowAccessModification,
                                          configuration.mergeInterfacesAggressively,
                                          classMergingHorizontalCounter));
        }

        if (classMergingVerticalCounter  .getCount() > 0 ||
            classMergingHorizontalCounter.getCount() > 0)
        {
            // Clean up inner class attributes to avoid loops.
            programClassPool.classesAccept(new RetargetedInnerClassAttributeRemover());

            // Update references to merged classes: first the referenced
            // classes, then the various actual descriptors.
            // Leave retargeted classes themselves unchanged and valid,
            // in case they aren't shrunk later on.
            programClassPool.classesAccept(new RetargetedClassFilter(null, new TargetClassChanger()));
            programClassPool.classesAccept(new RetargetedClassFilter(null, new ClassReferenceFixer(true)));
            programClassPool.classesAccept(new RetargetedClassFilter(null, new MemberReferenceFixer()));

            if (configuration.allowAccessModification)
            {
                // Fix the access flags of referenced merged classes and their
                // class members.
                programClassPool.classesAccept(
                    new AccessFixer());
            }

            // Fix the access flags of the inner classes information.
            programClassPool.classesAccept(
                new AllAttributeVisitor(
                new AllInnerClassesInfoVisitor(
                new InnerClassesAccessFixer())));

            // Tweak the descriptors of duplicate initializers, due to merged
            // parameter classes.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new DuplicateInitializerFixer(initializerFixCounter2)));

            if (initializerFixCounter2.getCount() > 0)
            {
                // Fix all invocations of tweaked initializers.
                programClassPool.classesAccept(
                    new AllMethodVisitor(
                    new AllAttributeVisitor(
                    new DuplicateInitializerInvocationFixer(addedCounter))));

                // Fix all references to tweaked initializers.
                programClassPool.classesAccept(new MemberReferenceFixer());
            }
        }

        if (methodInliningUnique)
        {
            // Inline methods that are only invoked once.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new MethodInliner(configuration.microEdition,
                                  configuration.allowAccessModification,
                                  true,
                                  methodInliningUniqueCounter))));
        }

        if (methodInliningShort)
        {
            // Inline short methods.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new MethodInliner(configuration.microEdition,
                                  configuration.allowAccessModification,
                                  false,
                                  methodInliningShortCounter))));
        }

        if (methodInliningTailrecursion)
        {
            // Simplify tail recursion calls.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new TailRecursionSimplifier(methodInliningTailrecursionCounter))));
        }

        if (fieldMarkingPrivate ||
            methodMarkingPrivate)
        {
            // Mark all class members that can not be made private.
            programClassPool.classesAccept(
                new NonPrivateMemberMarker());
        }

        if (fieldMarkingPrivate)
        {
            // Make all non-private fields private, whereever possible.
            programClassPool.classesAccept(
                new ClassAccessFilter(0, ClassConstants.ACC_INTERFACE,
                new AllFieldVisitor(
                new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                new MemberPrivatizer(fieldMarkingPrivateCounter)))));
        }

        if (methodMarkingPrivate)
        {
            // Make all non-private methods private, whereever possible.
            programClassPool.classesAccept(
                new ClassAccessFilter(0, ClassConstants.ACC_INTERFACE,
                new AllMethodVisitor(
                new MemberAccessFilter(0, ClassConstants.ACC_PRIVATE,
                new MemberPrivatizer(methodMarkingPrivateCounter)))));
        }

        if ((methodInliningUniqueCounter       .getCount() > 0 ||
             methodInliningShortCounter        .getCount() > 0 ||
             methodInliningTailrecursionCounter.getCount() > 0) &&
            configuration.allowAccessModification)
        {
            // Fix the access flags of referenced classes and class members,
            // for MethodInliner.
            programClassPool.classesAccept(
                new AccessFixer());
        }

        if (methodRemovalParameterCounter.getCount() > 0 ||
            classMergingVerticalCounter  .getCount() > 0 ||
            classMergingHorizontalCounter.getCount() > 0 ||
            methodMarkingPrivateCounter  .getCount() > 0 ||
            methodInliningUniqueCounter  .getCount() > 0 ||
            methodInliningShortCounter   .getCount() > 0)
        {
            // Fix invocations of interface methods, of methods that have become
            // non-abstract or private, and of methods that have moved to a
            // different package.
            programClassPool.classesAccept(
                new AllMemberVisitor(
                new AllAttributeVisitor(
                new MethodInvocationFixer())));
        }

        if (codeMerging)
        {
            // Share common blocks of code at branches.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new GotoCommonCodeReplacer(codeMergingCounter))));
        }

        // Create a branch target marker and a code attribute editor that can
        // be reused for all code attributes.
        BranchTargetFinder branchTargetFinder   = new BranchTargetFinder();
        CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor();

        InstructionSequenceConstants sequences =
            new InstructionSequenceConstants(programClassPool,
                                             libraryClassPool);

        List peepholeOptimizations = new ArrayList();
        if (codeSimplificationVariable)
        {
            // Peephole optimizations involving local variables.
            peepholeOptimizations.add(
                new InstructionSequencesReplacer(sequences.CONSTANTS,
                                                 sequences.VARIABLE,
                                                 branchTargetFinder, codeAttributeEditor, codeSimplificationVariableCounter));
        }

        if (codeSimplificationArithmetic)
        {
            // Peephole optimizations involving arithmetic operations.
            peepholeOptimizations.add(
                new InstructionSequencesReplacer(sequences.CONSTANTS,
                                                 sequences.ARITHMETIC,
                                                 branchTargetFinder, codeAttributeEditor, codeSimplificationArithmeticCounter));
        }

        if (codeSimplificationCast)
        {
            // Peephole optimizations involving cast operations.
            peepholeOptimizations.add(
                new InstructionSequencesReplacer(sequences.CONSTANTS,
                                                 sequences.CAST,
                                                 branchTargetFinder, codeAttributeEditor, codeSimplificationCastCounter));
        }

        if (codeSimplificationField)
        {
            // Peephole optimizations involving fields.
            peepholeOptimizations.add(
                new InstructionSequencesReplacer(sequences.CONSTANTS,
                                                 sequences.FIELD,
                                                 branchTargetFinder, codeAttributeEditor, codeSimplificationFieldCounter));
        }

        if (codeSimplificationBranch)
        {
            // Peephole optimizations involving branches.
            peepholeOptimizations.add(
                new InstructionSequencesReplacer(sequences.CONSTANTS,
                                                 sequences.BRANCH,
                                                 branchTargetFinder, codeAttributeEditor, codeSimplificationBranchCounter));

            // Include optimization of branches to branches and returns.
            peepholeOptimizations.add(
                new GotoGotoReplacer(codeAttributeEditor, codeSimplificationBranchCounter));
            peepholeOptimizations.add(
                new GotoReturnReplacer(codeAttributeEditor, codeSimplificationBranchCounter));
        }

        if (codeSimplificationString)
        {
            // Peephole optimizations involving branches.
            peepholeOptimizations.add(
                new InstructionSequencesReplacer(sequences.CONSTANTS,
                                                 sequences.STRING,
                                                 branchTargetFinder, codeAttributeEditor, codeSimplificationStringCounter));
        }

        if (!peepholeOptimizations.isEmpty())
        {
            // Convert the list into an array.
            InstructionVisitor[] peepholeOptimizationsArray =
                new InstructionVisitor[peepholeOptimizations.size()];
            peepholeOptimizations.toArray(peepholeOptimizationsArray);

            // Perform the peephole optimisations.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new PeepholeOptimizer(branchTargetFinder, codeAttributeEditor,
                new MultiInstructionVisitor(
                peepholeOptimizationsArray)))));
        }

        if (codeRemovalException)
        {
            // Remove unnecessary exception handlers.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new UnreachableExceptionRemover(codeRemovalExceptionCounter))));
        }

        if (codeRemovalSimple)
        {
            // Remove unreachable code.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new UnreachableCodeRemover(deletedCounter))));
        }

        if (codeRemovalVariable)
        {
            // Remove all unused local variables.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new VariableShrinker(codeRemovalVariableCounter))));
        }

        if (codeAllocationVariable)
        {
            // Optimize the variables.
            programClassPool.classesAccept(
                new AllMethodVisitor(
                new AllAttributeVisitor(
                new VariableOptimizer(false, codeAllocationVariableCounter))));
        }


        // Remove unused constants.
        programClassPool.classesAccept(
            new ConstantPoolShrinker());

        int classMarkingFinalCount            = classMarkingFinalCounter           .getCount();
        int classUnboxingEnumCount            = classUnboxingEnumCounter           .getCount();
        int classMergingVerticalCount         = classMergingVerticalCounter        .getCount();
        int classMergingHorizontalCount       = classMergingHorizontalCounter      .getCount();
        int fieldRemovalWriteonlyCount        = fieldRemovalWriteonlyCounter       .getCount();
        int fieldMarkingPrivateCount          = fieldMarkingPrivateCounter         .getCount();
        int fieldPropagationValueCount        = fieldPropagationValueCounter       .getCount();
        int methodMarkingPrivateCount         = methodMarkingPrivateCounter        .getCount();
        int methodMarkingStaticCount          = methodMarkingStaticCounter         .getCount();
        int methodMarkingFinalCount           = methodMarkingFinalCounter          .getCount();
        int methodRemovalParameterCount       = methodRemovalParameterCounter      .getCount() - methodMarkingStaticCounter.getCount() - initializerFixCounter1.getCount() - initializerFixCounter2.getCount();
        int methodPropagationParameterCount   = methodPropagationParameterCounter  .getCount();
        int methodPropagationReturnvalueCount = methodPropagationReturnvalueCounter.getCount();
        int methodInliningShortCount          = methodInliningShortCounter         .getCount();
        int methodInliningUniqueCount         = methodInliningUniqueCounter        .getCount();
        int methodInliningTailrecursionCount  = methodInliningTailrecursionCounter .getCount();
        int codeMergingCount                  = codeMergingCounter                 .getCount();
        int codeSimplificationVariableCount   = codeSimplificationVariableCounter  .getCount();
        int codeSimplificationArithmeticCount = codeSimplificationArithmeticCounter.getCount();
        int codeSimplificationCastCount       = codeSimplificationCastCounter      .getCount();
        int codeSimplificationFieldCount      = codeSimplificationFieldCounter     .getCount();
        int codeSimplificationBranchCount     = codeSimplificationBranchCounter    .getCount();
        int codeSimplificationStringCount     = codeSimplificationStringCounter    .getCount();
        int codeSimplificationAdvancedCount   = codeSimplificationAdvancedCounter  .getCount();
        int codeRemovalCount                  = deletedCounter                     .getCount() - addedCounter.getCount();
        int codeRemovalVariableCount          = codeRemovalVariableCounter         .getCount();
        int codeRemovalExceptionCount         = codeRemovalExceptionCounter        .getCount();
        int codeAllocationVariableCount       = codeAllocationVariableCounter      .getCount();

        // Forget about constant fields, parameters, and return values, if they
        // didn't lead to any useful optimizations. We want to avoid fruitless
        // additional optimization passes.
        if (codeSimplificationAdvancedCount == 0)
        {
            fieldPropagationValueCount        = 0;
            methodPropagationParameterCount   = 0;
            methodPropagationReturnvalueCount = 0;
        }

        if (configuration.verbose)
        {
            System.out.println("  Number of finalized classes:                 " + classMarkingFinalCount            + disabled(classMarkingFinal));
            System.out.println("  Number of unboxed enum classes:              " + classUnboxingEnumCount            + disabled(classUnboxingEnum));
            System.out.println("  Number of vertically merged classes:         " + classMergingVerticalCount         + disabled(classMergingVertical));
            System.out.println("  Number of horizontally merged classes:       " + classMergingHorizontalCount       + disabled(classMergingHorizontal));
            System.out.println("  Number of removed write-only fields:         " + fieldRemovalWriteonlyCount        + disabled(fieldRemovalWriteonly));
            System.out.println("  Number of privatized fields:                 " + fieldMarkingPrivateCount          + disabled(fieldMarkingPrivate));
            System.out.println("  Number of inlined constant fields:           " + fieldPropagationValueCount        + disabled(fieldPropagationValue));
            System.out.println("  Number of privatized methods:                " + methodMarkingPrivateCount         + disabled(methodMarkingPrivate));
            System.out.println("  Number of staticized methods:                " + methodMarkingStaticCount          + disabled(methodMarkingStatic));
            System.out.println("  Number of finalized methods:                 " + methodMarkingFinalCount           + disabled(methodMarkingFinal));
            System.out.println("  Number of removed method parameters:         " + methodRemovalParameterCount       + disabled(methodRemovalParameter));
            System.out.println("  Number of inlined constant parameters:       " + methodPropagationParameterCount   + disabled(methodPropagationParameter));
            System.out.println("  Number of inlined constant return values:    " + methodPropagationReturnvalueCount + disabled(methodPropagationReturnvalue));
            System.out.println("  Number of inlined short method calls:        " + methodInliningShortCount          + disabled(methodInliningShort));
            System.out.println("  Number of inlined unique method calls:       " + methodInliningUniqueCount         + disabled(methodInliningUnique));
            System.out.println("  Number of inlined tail recursion calls:      " + methodInliningTailrecursionCount  + disabled(methodInliningTailrecursion));
            System.out.println("  Number of merged code blocks:                " + codeMergingCount                  + disabled(codeMerging));
            System.out.println("  Number of variable peephole optimizations:   " + codeSimplificationVariableCount   + disabled(codeSimplificationVariable));
            System.out.println("  Number of arithmetic peephole optimizations: " + codeSimplificationArithmeticCount + disabled(codeSimplificationArithmetic));
            System.out.println("  Number of cast peephole optimizations:       " + codeSimplificationCastCount       + disabled(codeSimplificationCast));
            System.out.println("  Number of field peephole optimizations:      " + codeSimplificationFieldCount      + disabled(codeSimplificationField));
            System.out.println("  Number of branch peephole optimizations:     " + codeSimplificationBranchCount     + disabled(codeSimplificationBranch));
            System.out.println("  Number of string peephole optimizations:     " + codeSimplificationStringCount     + disabled(codeSimplificationString));
            System.out.println("  Number of simplified instructions:           " + codeSimplificationAdvancedCount   + disabled(codeSimplificationAdvanced));
            System.out.println("  Number of removed instructions:              " + codeRemovalCount                  + disabled(codeRemovalAdvanced));
            System.out.println("  Number of removed local variables:           " + codeRemovalVariableCount          + disabled(codeRemovalVariable));
            System.out.println("  Number of removed exception blocks:          " + codeRemovalExceptionCount         + disabled(codeRemovalException));
            System.out.println("  Number of optimized local variable frames:   " + codeAllocationVariableCount       + disabled(codeAllocationVariable));
        }

        return classMarkingFinalCount            > 0 ||
               classUnboxingEnumCount            > 0 ||
               classMergingVerticalCount         > 0 ||
               classMergingHorizontalCount       > 0 ||
               fieldRemovalWriteonlyCount        > 0 ||
               fieldMarkingPrivateCount          > 0 ||
               methodMarkingPrivateCount         > 0 ||
               methodMarkingStaticCount          > 0 ||
               methodMarkingFinalCount           > 0 ||
               fieldPropagationValueCount        > 0 ||
               methodRemovalParameterCount       > 0 ||
               methodPropagationParameterCount   > 0 ||
               methodPropagationReturnvalueCount > 0 ||
               methodInliningShortCount          > 0 ||
               methodInliningUniqueCount         > 0 ||
               methodInliningTailrecursionCount  > 0 ||
               codeMergingCount                  > 0 ||
               codeSimplificationVariableCount   > 0 ||
               codeSimplificationArithmeticCount > 0 ||
               codeSimplificationCastCount       > 0 ||
               codeSimplificationFieldCount      > 0 ||
               codeSimplificationBranchCount     > 0 ||
               codeSimplificationStringCount     > 0 ||
               codeSimplificationAdvancedCount   > 0 ||
               codeRemovalCount                  > 0 ||
               codeRemovalVariableCount          > 0 ||
               codeRemovalExceptionCount         > 0 ||
               codeAllocationVariableCount       > 0;
    }


    /**
     * Returns a String indicating whether the given flag is enabled or
     * disabled.
     */
    private String disabled(boolean flag)
    {
        return flag ? "" : "   (disabled)";
    }


    /**
     * Returns a String indicating whether the given flags are enabled or
     * disabled.
     */
    private String disabled(boolean flag1, boolean flag2)
    {
        return flag1 && flag2 ? "" :
               flag1 || flag2 ? "   (partially disabled)" :
                                "   (disabled)";
    }
}
