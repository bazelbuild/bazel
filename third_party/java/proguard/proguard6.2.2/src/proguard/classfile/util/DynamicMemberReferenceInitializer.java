/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.AttributeVisitor;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.editor.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.visitor.*;
import proguard.util.StringMatcher;

/**
 * This AttributeVisitor initializes any constant class member references of all
 * code that it visits. It currently handles invocations of
 *     Class#get[Declared]{Field,Constructor,Method} and
 *     Atomic{Integer,Long,Reference}FieldUpdater.newUpdater
 * with constant string arguments. It lets the corresponding string constants
 * refer to their class members in the program class pool or in the library
 * class pool. It may create new string constants and update the code, in order
 * to avoid clashes between identically named class members.
 * <p>
 * The class hierarchy and references must be initialized before using this
 * visitor.
 *
 * @see ClassSuperHierarchyInitializer
 * @see ClassReferenceInitializer
 *
 * @author Eric Lafortune
 */
public class DynamicMemberReferenceInitializer
extends      SimplifiedVisitor
implements   AttributeVisitor,
             InstructionVisitor,
             ConstantVisitor,
             MemberVisitor
{
    /*
    private static       boolean DEBUG = true;
    /*/
    private static final boolean DEBUG = false;
    //*/

    private static final int CLASS_INDEX       = InstructionSequenceMatcher.A;
    private static final int MEMBER_NAME_INDEX = InstructionSequenceMatcher.B;
    private static final int MEMBER_TYPE_INDEX = InstructionSequenceMatcher.C;


    private final ClassPool      programClassPool;
    private final ClassPool      libraryClassPool;
    private final WarningPrinter notePrinter;
    private final StringMatcher  noteFieldExceptionMatcher;
    private final StringMatcher  noteMethodExceptionMatcher;


    // Retrieving fields or methods from known, constant classes.
    private final InstructionSequenceMatcher knownItegerUpdaterMatcher;
    private final InstructionSequenceMatcher knownLongUpdaterMatcher;
    private final InstructionSequenceMatcher knownReferenceUpdaterMatcher;

    // Retrieving fields or methods from unknown classes.
    private final InstructionSequenceMatcher unknownIntegerUpdaterMatcher;
    private final InstructionSequenceMatcher unknownLongUpdaterMatcher;
    private final InstructionSequenceMatcher unknownReferenceUpdaterMatcher;

    private final MyDynamicMemberFinder dynamicMemberFinder = new MyDynamicMemberFinder();

    private final MemberFinder        memberFinder         = new MemberFinder(true);
    private final MemberFinder        declaredMemberFinder = new MemberFinder(false);
    private final CodeAttributeEditor codeAttributeEditor  = new CodeAttributeEditor();


    // Fields acting as parameters for the visitors.
    private Clazz referencedClass;


    /**
     * Creates a new DynamicMemberReferenceInitializer.
     */
    public DynamicMemberReferenceInitializer(ClassPool      programClassPool,
                                             ClassPool      libraryClassPool,
                                             WarningPrinter notePrinter,
                                             StringMatcher  noteFieldExceptionMatcher,
                                             StringMatcher  noteMethodExceptionMatcher)
    {
        this.programClassPool           = programClassPool;
        this.libraryClassPool           = libraryClassPool;
        this.notePrinter                = notePrinter;
        this.noteFieldExceptionMatcher  = noteFieldExceptionMatcher;
        this.noteMethodExceptionMatcher = noteMethodExceptionMatcher;

        // Create the instruction sequences and matchers.
        InstructionSequenceBuilder builder =
            new InstructionSequenceBuilder(programClassPool, libraryClassPool);

        // AtomicIntegerFieldUpdater.newUpdater(A.class, "someField").
        Instruction[] knownItegerUpdaterInstructions = builder
            .ldc_(CLASS_INDEX)
            .ldc_(MEMBER_NAME_INDEX)
            .invokestatic(ClassConstants.NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_INTEGER_FIELD_UPDATER,
                          ClassConstants.METHOD_NAME_NEW_UPDATER,
                          ClassConstants.METHOD_TYPE_NEW_INTEGER_UPDATER)
            .instructions();

        // AtomicLongFieldUpdater.newUpdater(A.class, "someField").
        Instruction[] knownLongUpdaterInstructions = builder
            .ldc_(CLASS_INDEX)
            .ldc_(MEMBER_NAME_INDEX)
            .invokestatic(ClassConstants.NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_LONG_FIELD_UPDATER,
                          ClassConstants.METHOD_NAME_NEW_UPDATER,
                          ClassConstants.METHOD_TYPE_NEW_LONG_UPDATER)
            .instructions();

        // AtomicReferenceFieldUpdater.newUpdater(A.class, B.class, "someField").
        Instruction[] knownReferenceUpdaterInstructions = builder
            .ldc_(CLASS_INDEX)
            .ldc_(MEMBER_TYPE_INDEX)
            .ldc_(MEMBER_NAME_INDEX)
            .invokestatic(ClassConstants.NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_REFERENCE_FIELD_UPDATER,
                          ClassConstants.METHOD_NAME_NEW_UPDATER,
                          ClassConstants.METHOD_TYPE_NEW_REFERENCE_UPDATER)
            .instructions();

        // AtomicIntegerFieldUpdater.newUpdater(..., "someField").
        Instruction[] unknownIntegerUpdaterInstructions = builder
            .ldc_(MEMBER_NAME_INDEX)
            .invokestatic(ClassConstants.NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_INTEGER_FIELD_UPDATER,
                          ClassConstants.METHOD_NAME_NEW_UPDATER,
                          ClassConstants.METHOD_TYPE_NEW_INTEGER_UPDATER)
            .instructions();

        // AtomicLongFieldUpdater.newUpdater(..., "someField").
        Instruction[] unknownLongUpdaterInstructions = builder
            .ldc_(MEMBER_NAME_INDEX)
            .invokestatic(ClassConstants.NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_LONG_FIELD_UPDATER,
                          ClassConstants.METHOD_NAME_NEW_UPDATER,
                          ClassConstants.METHOD_TYPE_NEW_LONG_UPDATER)
            .instructions();

        // AtomicReferenceFieldUpdater.newUpdater(..., "someField").
        final Instruction[] unknownReferenceUpdaterInstructions = builder
            .ldc_(MEMBER_NAME_INDEX)
            .invokestatic(ClassConstants.NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_REFERENCE_FIELD_UPDATER,
                          ClassConstants.METHOD_NAME_NEW_UPDATER,
                          ClassConstants.METHOD_TYPE_NEW_REFERENCE_UPDATER)
            .instructions();

        Constant[] constants = builder.constants();

        knownItegerUpdaterMatcher      = new InstructionSequenceMatcher(constants, knownItegerUpdaterInstructions);
        knownLongUpdaterMatcher        = new InstructionSequenceMatcher(constants, knownLongUpdaterInstructions);
        knownReferenceUpdaterMatcher   = new InstructionSequenceMatcher(constants, knownReferenceUpdaterInstructions);
        unknownIntegerUpdaterMatcher   = new InstructionSequenceMatcher(constants, unknownIntegerUpdaterInstructions);
        unknownLongUpdaterMatcher      = new InstructionSequenceMatcher(constants, unknownLongUpdaterInstructions);
        unknownReferenceUpdaterMatcher = new InstructionSequenceMatcher(constants, unknownReferenceUpdaterInstructions);
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        if (DEBUG)
        {
            System.out.println("DynamicMemberReferenceInitializer: "+clazz.getName()+"."+method.getName(clazz)+method.getDescriptor(clazz));
        }

        // Set up the code attribute editor.
        codeAttributeEditor.reset(codeAttribute.u4codeLength);

        // Find the dynamic class member references.
        codeAttribute.instructionsAccept(clazz, method, this);

        // Apply any changes to the code.
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        // Try to match get[Declared]{Field,Constructor,Method} constructs.
        instruction.accept(clazz, method, codeAttribute, offset, dynamicMemberFinder);

        // Try to match the AtomicIntegerFieldUpdater.newUpdater(
        // SomeClass.class, "someField") construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       knownItegerUpdaterMatcher,
                       unknownIntegerUpdaterMatcher, true, false, false,
                       "" + ClassConstants.TYPE_INT);

        // Try to match the AtomicLongFieldUpdater.newUpdater(
        // SomeClass.class, "someField") construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       knownLongUpdaterMatcher,
                       unknownLongUpdaterMatcher, true, false, false,
                       "" + ClassConstants.TYPE_LONG);

        // Try to match the AtomicReferenceFieldUpdater.newUpdater(
        // SomeClass.class, SomeClass.class, "someField") construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       knownReferenceUpdaterMatcher,
                       unknownReferenceUpdaterMatcher, true, false, false,
                       null);
    }


    /**
     * Tries to match the next instruction and fills out the string constant
     * or prints out a note accordingly.
     */
    private void matchGetMember(Clazz                      clazz,
                                Method                     method,
                                CodeAttribute              codeAttribute,
                                int                        offset,
                                Instruction                instruction,
                                InstructionSequenceMatcher constantSequenceMatcher,
                                InstructionSequenceMatcher variableSequenceMatcher,
                                boolean                    isField,
                                boolean                    isConstructor,
                                boolean                    isDeclared,
                                String                     memberDescriptor)
    {
        if (constantSequenceMatcher != null)
        {
            // Try to match the next instruction in the constant sequence.
            instruction.accept(clazz, method, codeAttribute, offset,
                               constantSequenceMatcher);

            // Did we find a match to fill out the string constant?
            if (constantSequenceMatcher.isMatching())
            {
                // Retrieve the offset of the instruction that loads the member
                // name. It's currently always the last but one instruction.
                int memberNameInstructionOffset =
                    constantSequenceMatcher.matchedInstructionOffset(
                    constantSequenceMatcher.instructionCount() - 2);

                // Get the member's class.
                int classIndex = constantSequenceMatcher.matchedConstantIndex(CLASS_INDEX);
                clazz.constantPoolEntryAccept(classIndex, this);

                if (referencedClass != null)
                {
                    // Get the field's type, if applicable.
                    int typeClassIndex = constantSequenceMatcher.matchedConstantIndex(MEMBER_TYPE_INDEX);
                    if (typeClassIndex > 0)
                    {
                        memberDescriptor =
                            ClassUtil.internalTypeFromClassName(clazz.getClassName(typeClassIndex));
                    }

                    // Get the member's name.
                    int memberNameIndex = constantSequenceMatcher.matchedConstantIndex(MEMBER_NAME_INDEX);
                    String memberName = clazz.getStringString(memberNameIndex);

                    // Create a new string constant and update the instruction.
                    initializeDynamicMemberReference(clazz,
                                                     memberNameInstructionOffset,
                                                     referencedClass,
                                                     memberName,
                                                     memberDescriptor,
                                                     isField,
                                                     isConstructor,
                                                     isDeclared);
                }

                // Don't look for the dynamic construct.
                variableSequenceMatcher.reset();
            }
        }

        // Try to match the next instruction in the variable sequence.
        instruction.accept(clazz, method, codeAttribute, offset,
                           variableSequenceMatcher);

        // Did we find a match to print out a note?
        if (variableSequenceMatcher.isMatching())
        {
            int memberNameIndex = variableSequenceMatcher.matchedConstantIndex(MEMBER_NAME_INDEX);
            String memberName = clazz.getStringString(memberNameIndex);

            // Print out a note about the dynamic invocation.
            printDynamicMemberAccessNote(clazz,
                                         memberName,
                                         memberDescriptor,
                                         isField,
                                         isConstructor,
                                         isDeclared);
        }
    }


    // Implementations for ConstantVisitor.

    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Remember the referenced class.
        referencedClass = ClassUtil.isInternalArrayType(classConstant.getName(clazz)) ?
            null :
            classConstant.referencedClass;
    }


    // Small utility methods.

    /**
     * Creates a new string constant for the specified referenced class member,
     * and updates the instruction that loads it.
     */
    private void initializeDynamicMemberReference(Clazz   clazz,
                                                  int     memberNameInstructionOffset,
                                                  Clazz   referencedClass,
                                                  String  memberName,
                                                  String  memberDescriptor,
                                                  boolean isField,
                                                  boolean isConstructor,
                                                  boolean isDeclared)
    {
        // See if we can find the referenced class member locally, or
        // somewhere in the hierarchy.
        MemberFinder referencedMemberFinder = isDeclared ?
            declaredMemberFinder :
            memberFinder;

        Member referencedMember =
            referencedMemberFinder.findMember(referencedClass,
                                              memberName,
                                              memberDescriptor,
                                              isField);

        if (DEBUG)
        {
            System.out.println("DynamicMemberReferenceInitializer: ["+clazz.getName()+"] matched string ["+memberName+"]: in ["+referencedClass+"] -> ["+referencedMember+"]");
        }

        if (referencedMember != null)
        {
            if (!isDeclared)
            {
                referencedClass = referencedMemberFinder.correspondingClass();
            }

            // Update the string constant.
            //stringConstant.referencedMember = referencedMember;
            //stringConstant.referencedClass  = referencedClass;

            // Create a new string constant with the found references.
            int stringConstantIndex =
                new ConstantPoolEditor((ProgramClass)clazz).addStringConstant(memberName,
                                                                              referencedClass,
                                                                              referencedMember);

            // Update the instruction.
            codeAttributeEditor.replaceInstruction(memberNameInstructionOffset,
                                                   new ConstantInstruction(InstructionConstants.OP_LDC,
                                                                           stringConstantIndex));
        }
    }


    /**
     * Prints out a note on the matched dynamic invocation of a constructor.
     */
    private void printDynamicConstructorAccessNote(Clazz   clazz,
                                                   Clazz   referencedClass,
                                                   String  memberDescriptor,
                                                   boolean isDeclared)
    {
        // Print out a note about the dynamic invocation.
        if (notePrinter != null &&
            notePrinter.accepts(clazz.getName()))
        {
            // Is the class member name in the list of exceptions?
            if (noteMethodExceptionMatcher == null ||
                !noteMethodExceptionMatcher.matches(ClassConstants.METHOD_NAME_INIT))
            {
                // Print out the actual note.
                notePrinter.print(clazz.getName(),
                                  "Note: " +
                                  ClassUtil.externalClassName(clazz.getName()) +
                                  " retrieves a " +
                                  (isDeclared    ? "declared " : "") +
                                  "constructor '" +
                                  ClassConstants.METHOD_NAME_INIT +
                                  JavaConstants.METHOD_ARGUMENTS_OPEN +
                                  ClassUtil.externalMethodArguments(memberDescriptor) +
                                  JavaConstants.METHOD_ARGUMENTS_CLOSE +
                                  "' dynamically");

                // Print out notes about potential candidates.
                ClassVisitor classVisitor =
                    new AllMethodVisitor(
                    new MemberNameFilter(ClassConstants.METHOD_NAME_INIT,
                    new MemberDescriptorFilter(memberDescriptor, this)));

                if (referencedClass != null)
                {
                    referencedClass.hierarchyAccept(true, !isDeclared, false, false,
                                                    classVisitor);
                }
                else
                {
                    programClassPool.classesAcceptAlphabetically(classVisitor);
                    libraryClassPool.classesAcceptAlphabetically(classVisitor);
                }
            }
        }
    }


    /**
     * Prints out a note on the matched dynamic access to a class member.
     */
    private void printDynamicMemberAccessNote(Clazz   clazz,
                                              String  memberName,
                                              String  memberDescriptor,
                                              boolean isField,
                                              boolean isConstructor,
                                              boolean isDeclared)
    {
        // Print out a note about the dynamic invocation.
        if (notePrinter != null &&
            notePrinter.accepts(clazz.getName()))
        {
            // Is the class member name in the list of exceptions?
            StringMatcher noteExceptionMatcher = isField ?
                noteFieldExceptionMatcher :
                noteMethodExceptionMatcher;

            if (noteExceptionMatcher == null ||
                !noteExceptionMatcher.matches(memberName))
            {
                // Compose the external member name and partial descriptor.
                String externalMemberDescription = memberName;

                if (!isField)
                {
                    externalMemberDescription +=
                        JavaConstants.METHOD_ARGUMENTS_OPEN +
                        ClassUtil.externalMethodArguments(memberDescriptor) +
                        JavaConstants.METHOD_ARGUMENTS_CLOSE;
                }

                // Print out the actual note.
                notePrinter.print(clazz.getName(),
                                  "Note: " +
                                  ClassUtil.externalClassName(clazz.getName()) +
                                  " accesses a " +
                                  (isDeclared    ? "declared " : "") +
                                  (isField       ? "field"       :
                                   isConstructor ? "constructor" :
                                                   "method") +
                                  " '" +
                                  externalMemberDescription +
                                  "' dynamically");

                // Print out notes about potential candidates.
                ClassVisitor classVisitor;

                if (isField)
                {
                    classVisitor = memberDescriptor == null ?
                       new AllFieldVisitor(
                       new MemberNameFilter(memberName, this)) :
                       new AllFieldVisitor(
                       new MemberNameFilter(memberName,
                       new MemberDescriptorFilter(memberDescriptor, this)));
                }
                else
                {
                    classVisitor =
                        new AllMethodVisitor(
                        new MemberNameFilter(memberName,
                        new MemberDescriptorFilter(memberDescriptor, this)));
                }

                programClassPool.classesAcceptAlphabetically(classVisitor);
                libraryClassPool.classesAcceptAlphabetically(classVisitor);
            }
        }
    }


    // Implementations for MemberVisitor.

    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        if (notePrinter.accepts(programClass.getName()))
        {
            System.out.println("      Maybe this is program field '" +
                               ClassUtil.externalFullClassDescription(0, programClass.getName()) +
                               " { " +
                               ClassUtil.externalFullFieldDescription(0, programField.getName(programClass), programField.getDescriptor(programClass)) +
                               "; }'");
        }
    }


    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        if (notePrinter.accepts(programClass.getName()))
        {
            System.out.println("      Maybe this is program method '" +
                               ClassUtil.externalFullClassDescription(0, programClass.getName()) +
                               " { " +
                               ClassUtil.externalFullMethodDescription(programClass.getName(), 0, programMethod.getName(programClass), programMethod.getDescriptor(programClass)) +
                               "; }'");
        }
    }


    public void visitLibraryField(LibraryClass libraryClass, LibraryField libraryField)
    {
        if (notePrinter.accepts(libraryClass.getName()))
        {
            System.out.println("      Maybe this is library field '" +
                               ClassUtil.externalFullClassDescription(0, libraryClass.getName()) +
                               " { " +
                               ClassUtil.externalFullFieldDescription(0, libraryField.getName(libraryClass), libraryField.getDescriptor(libraryClass)) +
                               "; }'");
        }
    }


    public void visitLibraryMethod(LibraryClass libraryClass, LibraryMethod libraryMethod)
    {
        if (notePrinter.accepts(libraryClass.getName()))
        {
            System.out.println("      Maybe this is library method '" +
                               ClassUtil.externalFullClassDescription(0, libraryClass.getName()) +
                               " { " +
                               ClassUtil.externalFullMethodDescription(libraryClass.getName(), 0, libraryMethod.getName(libraryClass), libraryMethod.getDescriptor(libraryClass)) +
                               "; }'");
        }
    }


    /**
     * This InstructionVisitor finds get[Declared]{Field,Constructor,Method}
     * constructs with constant arguments. It then makes sure the class member
     * name strings point to the class members, or it prints out notes about the
     * possible alternatives.
     */
    private class MyDynamicMemberFinder
    extends       SimplifiedVisitor
    implements    InstructionVisitor,
                  ConstantVisitor
    {
        private static final int LABEL_START                 =  0; // ldc SomeClass.class
        private static final int LABEL_LOAD_MEMBER_NAME      =  1; // [ ldc "someMethod"/"someField" ]
        private static final int LABEL_LOAD_CLASS_ARRAY_SIZE =  2; // [ sipush #someParameterCount
        private static final int LABEL_CREATE_CLASS_ARRAY    =  3; //   anewarray java/lang/Class
        private static final int LABEL_DUP_CLASS_ARRAY       =  4; //   [ dup
        private static final int LABEL_LOAD_PARAMETER_INDEX  =  5; //     sipush #someParameterIndex
        private static final int LABEL_LOAD_PARAMETER_TYPE   =  6; //     ldc SomeParameterClass.class / getstatic java/lang/SomePrimitive.TYPE
        private static final int LABEL_STORE_PARAMETER       =  7; //     aastore ]* ] / aconst_null
        private static final int LABEL_GET_MEMBER            =  8; // invokevirtual java/lang/Class.getField/getConstructor/getMethod

        private int          label;
        private int          instructionOffset;
        private int          memberNameInstructionOffset;
        private Clazz        referencedClass;
        private String       memberName;
        private int          parameterCount;
        private int          parameterIndex;
        private StringBuffer parameterTypes = new StringBuffer();


        public void reset()
        {
            label           = LABEL_START;
            referencedClass = null;
            memberName      = null;
            parameterTypes.setLength(0);
        }


        // Implementations for InstructionVisitor.

        public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
        {
            if (DEBUG)
            {
                System.out.println("Label ["+label+"] A "+instruction.toString(offset));
            }

            reset();
        }


        public void visitSimpleInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, SimpleInstruction simpleInstruction)
        {
            if (DEBUG)
            {
                System.out.println("Label ["+label+"] S "+simpleInstruction.toString(offset));
            }

            int transition = label | simpleInstruction.canonicalOpcode() << 8;

            switch (transition)
            {
                case LABEL_START               | InstructionConstants.OP_ICONST_0 << 8:
                case LABEL_LOAD_MEMBER_NAME    | InstructionConstants.OP_ICONST_0 << 8:
                case LABEL_CREATE_CLASS_ARRAY  | InstructionConstants.OP_ICONST_0 << 8:
                case LABEL_DUP_CLASS_ARRAY     | InstructionConstants.OP_ICONST_0 << 8:
                case LABEL_LOAD_PARAMETER_TYPE | InstructionConstants.OP_ICONST_0 << 8:
                case LABEL_STORE_PARAMETER     | InstructionConstants.OP_ICONST_0 << 8:
                case LABEL_GET_MEMBER          | InstructionConstants.OP_ICONST_0 << 8:
                    // This could be the start of creating a class array.
                    reset();
                    parameterCount = simpleInstruction.constant;
                    label          = LABEL_CREATE_CLASS_ARRAY;
                    break;

                case LABEL_LOAD_CLASS_ARRAY_SIZE | InstructionConstants.OP_ICONST_0 << 8:
                    parameterCount = simpleInstruction.constant;
                    label          = LABEL_CREATE_CLASS_ARRAY;
                    break;

                case LABEL_LOAD_CLASS_ARRAY_SIZE | InstructionConstants.OP_ACONST_NULL << 8:
                    parameterCount = 0;
                    label          = LABEL_GET_MEMBER;
                    break;

                case LABEL_DUP_CLASS_ARRAY | InstructionConstants.OP_DUP << 8:
                    label = LABEL_LOAD_PARAMETER_INDEX;
                    break;

                case LABEL_LOAD_PARAMETER_INDEX | InstructionConstants.OP_ICONST_0 << 8:
                    // Is it pushing the expected parameter index?
                    if (parameterIndex == simpleInstruction.constant)
                    {
                        label = LABEL_LOAD_PARAMETER_TYPE;
                    }
                    else
                    {
                        // This could be the start of creating a class array.
                        reset();
                        parameterCount = simpleInstruction.constant;
                        label          = LABEL_CREATE_CLASS_ARRAY;
                    }
                    break;

                case LABEL_STORE_PARAMETER | InstructionConstants.OP_AASTORE << 8:
                    // Are we still expecting more parameters?
                    label = ++parameterIndex < parameterCount ?
                        LABEL_DUP_CLASS_ARRAY :
                        LABEL_GET_MEMBER;
                    break;

                default:
                    reset();
                    break;
            }
        }


        public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
        {
            if (DEBUG)
            {
                System.out.println("Label ["+label+"] C "+constantInstruction.toString(offset));
            }

            // Let the constant figure out the transition.
            switch (constantInstruction.canonicalOpcode())
            {
                case InstructionConstants.OP_LDC:
                    instructionOffset = offset;
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                    break;

                case InstructionConstants.OP_GETSTATIC:
                case InstructionConstants.OP_INVOKEVIRTUAL:
                    clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                    break;

                case InstructionConstants.OP_ANEWARRAY:
                    if (label == LABEL_CREATE_CLASS_ARRAY)
                    {
                        clazz.constantPoolEntryAccept(constantInstruction.constantIndex, this);
                    }
                    else
                    {
                        reset();
                    }
                    break;

                default:
                    reset();
                    break;
            }
        }


        // Implementations for ConstantVisitor.

        public void visitAnyConstant(Clazz clazz, Constant constant)
        {
            reset();
        }


        public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
        {
            // Argument of ldc or anewarray instruction.
            switch (label)
            {
                case LABEL_START:
                    referencedClass = classConstant.referencedClass;

                    label = LABEL_LOAD_MEMBER_NAME;
                    break;

                case LABEL_CREATE_CLASS_ARRAY:
                    if (classConstant.getName(clazz).equals(ClassConstants.NAME_JAVA_LANG_CLASS))
                    {
                        parameterIndex = 0;
                        label = parameterCount > 0 ?
                            LABEL_DUP_CLASS_ARRAY :
                            LABEL_GET_MEMBER;
                    }
                    else
                    {
                        referencedClass = classConstant.referencedClass;

                        label = LABEL_LOAD_MEMBER_NAME;
                    }
                    break;

                case LABEL_LOAD_PARAMETER_TYPE:
                    String parameterType =
                        ClassUtil.internalTypeFromClassType(classConstant.getName(clazz));

                    parameterTypes.append(parameterType);

                    label = LABEL_STORE_PARAMETER;
                    break;

                default:
                    // For other states, we'll treat this as a potential
                    // initial class name.
                    referencedClass = classConstant.referencedClass;

                    label = LABEL_LOAD_MEMBER_NAME;
                    break;
            }
        }


        public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
        {
            // Argument of ldc instruction.
            switch (label)
            {
                case LABEL_LOAD_MEMBER_NAME:
                    break;

                default:
                    // For other states, we'll treat this as a potential
                    // initial method name, without a known class.
                    referencedClass = null;
                    break;
            }

            // Whatever state, we'll treat this as a potential method name.
            memberNameInstructionOffset = instructionOffset;
            memberName                  = stringConstant.getString(clazz);

            label = LABEL_LOAD_CLASS_ARRAY_SIZE;
        }


        public void visitFieldrefConstant(Clazz clazz, FieldrefConstant fieldrefConstant)
        {
            // Argument of getstatic instruction.
            switch (label)
            {
                case LABEL_LOAD_PARAMETER_TYPE:
                    String className = fieldrefConstant.getClassName(clazz);
                    String fieldName = fieldrefConstant.getName(clazz);
                    String fieldType = fieldrefConstant.getType(clazz);

                    if (className.startsWith(ClassConstants.PACKAGE_JAVA_LANG) &&
                        fieldName.equals(ClassConstants.FIELD_NAME_TYPE)       &&
                        fieldType.equals(ClassConstants.FIELD_TYPE_TYPE))
                    {
                        char parameterType =
                            ClassUtil.internalPrimitiveTypeFromNumericClassName(className);

                        parameterTypes.append(parameterType);

                        label = LABEL_STORE_PARAMETER;
                    }
                    else
                    {
                        reset();
                    }
                    break;

                default:
                    reset();
                    break;
            }
        }


        public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
        {
            // Argument of invokevirtual instruction.
            String className = methodrefConstant.getClassName(clazz);

            if (className.equals(ClassConstants.NAME_JAVA_LANG_CLASS))
            {
                String methodName = methodrefConstant.getName(clazz);
                String methodType = methodrefConstant.getType(clazz);

                if (label == LABEL_LOAD_CLASS_ARRAY_SIZE &&
                    methodType.equals(ClassConstants.METHOD_TYPE_CLASS_GET_FIELD) &&
                    memberName != null)
                {
                    if (methodName.equals(ClassConstants.METHOD_NAME_CLASS_GET_FIELD))
                    {
                        resolveMemberString(clazz, true, false, false);
                    }
                    else if (methodName.equals(ClassConstants.METHOD_NAME_CLASS_GET_DECLARED_FIELD))
                    {
                        resolveMemberString(clazz, true, false, true);
                    }
                    else
                    {
                        reset();
                    }
                }
                else if (label == LABEL_GET_MEMBER &&
                         methodType.equals(ClassConstants.METHOD_TYPE_CLASS_GET_CONSTRUCTOR))
                {
                    if (methodName.equals(ClassConstants.METHOD_NAME_CLASS_GET_CONSTRUCTOR))
                    {
                        resolveMemberString(clazz, false, true, false);
                    }
                    else if (methodName.equals(ClassConstants.METHOD_NAME_CLASS_GET_DECLARED_CONSTRUCTOR))
                    {
                        resolveMemberString(clazz, false, true, true);
                    }
                    else
                    {
                        reset();
                    }
                }
                else if (label == LABEL_GET_MEMBER &&
                         methodType.equals(ClassConstants.METHOD_TYPE_CLASS_GET_METHOD) &&
                         memberName != null)
                {
                    if (methodName.equals(ClassConstants.METHOD_NAME_CLASS_GET_METHOD))
                    {
                        resolveMemberString(clazz, false, false, false);
                    }
                    else if (methodName.equals(ClassConstants.METHOD_NAME_CLASS_GET_DECLARED_METHOD))
                    {
                        resolveMemberString(clazz, false, false, true);
                    }
                    else
                    {
                        reset();
                    }
                }
                else
                {
                    reset();
                }
            }
            else
            {
                reset();
            }
        }


        /**
         * Links the referenced class member in the string, or prints out
         * notes about the possible alternatives.
         */
        private void resolveMemberString(Clazz   clazz,
                                         boolean isField,
                                         boolean isConstructor,
                                         boolean isDeclared)
        {
            String memberDescriptor = isField ?
                null :
                ClassConstants.METHOD_ARGUMENTS_OPEN +
                parameterTypes.toString() +
                ClassConstants.METHOD_ARGUMENTS_CLOSE +
                "L***;";

            if (DEBUG)
            {
                System.out.println("DynamicMemberReferenceInitializer: found member access");
                System.out.println("  isField           = "+isField);
                System.out.println("  isConstructor     = "+isConstructor);
                System.out.println("  isDeclared        = "+isDeclared);
                System.out.println("  referenced class  = "+(referencedClass  == null ? "(none)" : "["+referencedClass.getName()+"]"));
                System.out.println("  member name       = "+(memberName       == null ? "(none)" : "["+memberName+"]"));
                System.out.println("  member descriptor = "+(memberDescriptor == null ? "(none)" : "["+memberDescriptor+"]"));
            }

            if (referencedClass != null)
            {
                if (isConstructor)
                {
                    // We currently can't fill out some reference to a
                    // constructor. Just print out notes instead.
                    printDynamicConstructorAccessNote(clazz,
                                                      referencedClass,
                                                      memberDescriptor,
                                                      isDeclared);
                }
                else
                {
                    // Create a new string constant and update the instruction.
                    initializeDynamicMemberReference(clazz,
                                                     memberNameInstructionOffset,
                                                     referencedClass,
                                                     memberName,
                                                     memberDescriptor,
                                                     isField,
                                                     isConstructor,
                                                     isDeclared);
                }
            }
            else
            {
                // Print out notes about the method in some unknown class.
                printDynamicMemberAccessNote(clazz,
                                             isConstructor ?
                                                 ClassConstants.METHOD_NAME_INIT :
                                                 memberName,
                                             memberDescriptor,
                                             isField,
                                             isConstructor,
                                             isDeclared);
            }
        }
    }
}