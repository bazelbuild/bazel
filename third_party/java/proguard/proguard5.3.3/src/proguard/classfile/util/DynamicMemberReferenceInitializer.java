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
package proguard.classfile.util;

import proguard.classfile.*;
import proguard.classfile.attribute.CodeAttribute;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.visitor.*;
import proguard.util.StringMatcher;

/**
 * This InstructionVisitor initializes any constant
 * <code>Class.get[Declared]{Field,Method}</code> references of all instructions
 * it visits. More specifically, it fills out the references of string constant
 * pool entries that refer to a class member in the program class pool or in the
 * library class pool.
 * <p>
 * It optionally prints notes if on usage of
 * <code>(SomeClass)Class.forName(variable).newInstance()</code>.
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
implements   InstructionVisitor,
             ConstantVisitor,
             MemberVisitor
{
    //*
    private static final boolean DEBUG = false;
    /*/
    private static       boolean DEBUG = true;
    //*/

    public static final int CLASS_INDEX       = InstructionSequenceMatcher.X;
    public static final int MEMBER_NAME_INDEX = InstructionSequenceMatcher.Y;
    public static final int TYPE_CLASS_INDEX  = InstructionSequenceMatcher.Z;

    public static final int PARAMETER0_CLASS_INDEX = InstructionSequenceMatcher.A;
    public static final int PARAMETER1_CLASS_INDEX = InstructionSequenceMatcher.B;
    public static final int PARAMETER2_CLASS_INDEX = InstructionSequenceMatcher.C;
    public static final int PARAMETER3_CLASS_INDEX = InstructionSequenceMatcher.D;


    private final Constant[] GET_FIELD_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(1, 2, null, null),
        new ClassConstant(3, null),
        new NameAndTypeConstant(4, 5),
        new Utf8Constant(ClassConstants.NAME_JAVA_LANG_CLASS),
        new Utf8Constant(ClassConstants.METHOD_NAME_CLASS_GET_FIELD),
        new Utf8Constant(ClassConstants.METHOD_TYPE_CLASS_GET_FIELD),
    };

    private final Constant[] GET_DECLARED_FIELD_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(1, 2, null, null),
        new ClassConstant(3, null),
        new NameAndTypeConstant(4, 5),
        new Utf8Constant(ClassConstants.NAME_JAVA_LANG_CLASS),
        new Utf8Constant(ClassConstants.METHOD_NAME_CLASS_GET_DECLARED_FIELD),
        new Utf8Constant(ClassConstants.METHOD_TYPE_CLASS_GET_DECLARED_FIELD),
    };

    private final Constant[] GET_CONSTRUCTOR_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(1, 2, null, null),
        new ClassConstant(3, null),
        new NameAndTypeConstant(4, 5),
        new Utf8Constant(ClassConstants.NAME_JAVA_LANG_CLASS),
        new Utf8Constant(ClassConstants.CONSTRUCTOR_NAME_CLASS_GET_CONSTRUCTOR),
        new Utf8Constant(ClassConstants.CONSTRUCTOR_TYPE_CLASS_GET_CONSTRUCTOR),
    };

    private final Constant[] GET_DECLARED_CONSTRUCTOR_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(1, 2, null, null),
        new ClassConstant(3, null),
        new NameAndTypeConstant(4, 5),
        new Utf8Constant(ClassConstants.NAME_JAVA_LANG_CLASS),
        new Utf8Constant(ClassConstants.CONSTRUCTOR_NAME_CLASS_GET_DECLARED_CONSTRUCTOR),
        new Utf8Constant(ClassConstants.CONSTRUCTOR_TYPE_CLASS_GET_DECLARED_CONSTRUCTOR),
    };

    private final Constant[] GET_METHOD_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(1, 2, null, null),
        new ClassConstant(3, null),
        new NameAndTypeConstant(4, 5),
        new Utf8Constant(ClassConstants.NAME_JAVA_LANG_CLASS),
        new Utf8Constant(ClassConstants.METHOD_NAME_CLASS_GET_METHOD),
        new Utf8Constant(ClassConstants.METHOD_TYPE_CLASS_GET_METHOD),
    };

    private final Constant[] GET_DECLARED_METHOD_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(1, 2, null, null),
        new ClassConstant(3, null),
        new NameAndTypeConstant(4, 5),
        new Utf8Constant(ClassConstants.NAME_JAVA_LANG_CLASS),
        new Utf8Constant(ClassConstants.METHOD_NAME_CLASS_GET_DECLARED_METHOD),
        new Utf8Constant(ClassConstants.METHOD_TYPE_CLASS_GET_DECLARED_METHOD),
    };

    private final Constant[] NEW_INTEGER_UPDATER_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(1, 2, null, null),
        new ClassConstant(3, null),
        new NameAndTypeConstant(4, 5),
        new Utf8Constant(ClassConstants.NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_INTEGER_FIELD_UPDATER),
        new Utf8Constant(ClassConstants.METHOD_NAME_NEW_UPDATER),
        new Utf8Constant(ClassConstants.METHOD_TYPE_NEW_INTEGER_UPDATER),
    };

    private final Constant[] NEW_LONG_UPDATER_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(1, 2, null, null),
        new ClassConstant(3, null),
        new NameAndTypeConstant(4, 5),
        new Utf8Constant(ClassConstants.NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_LONG_FIELD_UPDATER),
        new Utf8Constant(ClassConstants.METHOD_NAME_NEW_UPDATER),
        new Utf8Constant(ClassConstants.METHOD_TYPE_NEW_LONG_UPDATER),
    };

    private final Constant[] NEW_REFERENCE_UPDATER_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(1, 2, null, null),
        new ClassConstant(3, null),
        new NameAndTypeConstant(4, 5),
        new Utf8Constant(ClassConstants.NAME_JAVA_UTIL_CONCURRENT_ATOMIC_ATOMIC_REFERENCE_FIELD_UPDATER),
        new Utf8Constant(ClassConstants.METHOD_NAME_NEW_UPDATER),
        new Utf8Constant(ClassConstants.METHOD_TYPE_NEW_REFERENCE_UPDATER),
    };

    // SomeClass.class.get[Declared]Field("someField").
    private final Instruction[] CONSTANT_GET_FIELD_INSTRUCTIONS = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, CLASS_INDEX),
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
    };

//    // SomeClass.class.get[Declared]Constructor(new Class[] {}).
//    private final Instruction[] CONSTANT_GET_CONSTRUCTOR_INSTRUCTIONS0 = new Instruction[]
//    {
//        new ConstantInstruction(InstructionConstants.OP_LDC, CLASS_INDEX),
//        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
//        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
//        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
//    };
//
//    // SomeClass.class.get[Declared]Constructor(new Class[] { A.class }).
//    private final Instruction[] CONSTANT_GET_CONSTRUCTOR_INSTRUCTIONS1 = new Instruction[]
//    {
//        new ConstantInstruction(InstructionConstants.OP_LDC, CLASS_INDEX),
//        new SimpleInstruction(InstructionConstants.OP_ICONST_1),
//        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
//        new SimpleInstruction(InstructionConstants.OP_DUP),
//        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
//        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER0_CLASS_INDEX),
//        new SimpleInstruction(InstructionConstants.OP_AASTORE),
//        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
//    };
//
//    // SomeClass.class.get[Declared]Constructor(new Class[] { A.class, B.class }).
//    private final Instruction[] CONSTANT_GET_CONSTRUCTOR_INSTRUCTIONS2 = new Instruction[]
//    {
//        new ConstantInstruction(InstructionConstants.OP_LDC, CLASS_INDEX),
//        new SimpleInstruction(InstructionConstants.OP_ICONST_2),
//        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
//        new SimpleInstruction(InstructionConstants.OP_DUP),
//        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
//        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER0_CLASS_INDEX),
//        new SimpleInstruction(InstructionConstants.OP_AASTORE),
//        new SimpleInstruction(InstructionConstants.OP_DUP),
//        new SimpleInstruction(InstructionConstants.OP_ICONST_1),
//        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER1_CLASS_INDEX),
//        new SimpleInstruction(InstructionConstants.OP_AASTORE),
//        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
//    };

    // SomeClass.class.get[Declared]Method("someMethod", new Class[] {}).
    private final Instruction[] CONSTANT_GET_METHOD_INSTRUCTIONS0 = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, CLASS_INDEX),
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
    };

    // SomeClass.class.get[Declared]Method("someMethod", new Class[] { A.class }).
    private final Instruction[] CONSTANT_GET_METHOD_INSTRUCTIONS1 = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, CLASS_INDEX),
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new SimpleInstruction(InstructionConstants.OP_ICONST_1),
        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
        new SimpleInstruction(InstructionConstants.OP_DUP),
        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER0_CLASS_INDEX),
        new SimpleInstruction(InstructionConstants.OP_AASTORE),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
    };

    // SomeClass.class.get[Declared]Method("someMethod", new Class[] { A.class, B.class }).
    private final Instruction[] CONSTANT_GET_METHOD_INSTRUCTIONS2 = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, CLASS_INDEX),
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new SimpleInstruction(InstructionConstants.OP_ICONST_2),
        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
        new SimpleInstruction(InstructionConstants.OP_DUP),
        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER0_CLASS_INDEX),
        new SimpleInstruction(InstructionConstants.OP_AASTORE),
        new SimpleInstruction(InstructionConstants.OP_DUP),
        new SimpleInstruction(InstructionConstants.OP_ICONST_1),
        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER1_CLASS_INDEX),
        new SimpleInstruction(InstructionConstants.OP_AASTORE),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
    };

    // AtomicIntegerFieldUpdater.newUpdater(A.class, "someField").
    // AtomicLongFieldUpdater.newUpdater(A.class, "someField").
    private final Instruction[] CONSTANT_NEW_PRIMITIVE_UPDATER_INSTRUCTIONS = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, CLASS_INDEX),
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, 0),
    };

    // AtomicReferenceFieldUpdater.newUpdater(A.class, B.class, "someField").
    private final Instruction[] CONSTANT_NEW_REFERENCE_UPDATER_INSTRUCTIONS = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, CLASS_INDEX),
        new ConstantInstruction(InstructionConstants.OP_LDC, TYPE_CLASS_INDEX),
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, 0),
    };

    // get[Declared]Field("someField").
    private final Instruction[] GET_FIELD_INSTRUCTIONS = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
    };

//    // get[Declared]Constructor(new Class[] {}).
//    private final Instruction[] GET_CONSTRUCTOR_INSTRUCTIONS0 = new Instruction[]
//    {
//        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
//        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
//        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
//    };

    // get[Declared]Constructor(new Class[] { A.class }).
    private final Instruction[] GET_CONSTRUCTOR_INSTRUCTIONS1 = new Instruction[]
    {
        new SimpleInstruction(InstructionConstants.OP_ICONST_1),
        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
        new SimpleInstruction(InstructionConstants.OP_DUP),
        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER0_CLASS_INDEX),
        new SimpleInstruction(InstructionConstants.OP_AASTORE),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
    };

    // get[Declared]Constructor(new Class[] { A.class, B.class }).
    private final Instruction[] GET_CONSTRUCTOR_INSTRUCTIONS2 = new Instruction[]
    {
        new SimpleInstruction(InstructionConstants.OP_ICONST_2),
        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
        new SimpleInstruction(InstructionConstants.OP_DUP),
        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER0_CLASS_INDEX),
        new SimpleInstruction(InstructionConstants.OP_AASTORE),
        new SimpleInstruction(InstructionConstants.OP_DUP),
        new SimpleInstruction(InstructionConstants.OP_ICONST_1),
        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER1_CLASS_INDEX),
        new SimpleInstruction(InstructionConstants.OP_AASTORE),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
    };

    // get[Declared]Method("someMethod", new Class[] {}).
    private final Instruction[] GET_METHOD_INSTRUCTIONS0 = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
    };

    // get[Declared]Method("someMethod", new Class[] { A.class }).
    private final Instruction[] GET_METHOD_INSTRUCTIONS1 = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new SimpleInstruction(InstructionConstants.OP_ICONST_1),
        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
        new SimpleInstruction(InstructionConstants.OP_DUP),
        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER0_CLASS_INDEX),
        new SimpleInstruction(InstructionConstants.OP_AASTORE),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
    };

    // get[Declared]Method("someMethod", new Class[] { A.class, B.class }).
    private final Instruction[] GET_METHOD_INSTRUCTIONS2 = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new SimpleInstruction(InstructionConstants.OP_ICONST_2),
        new ConstantInstruction(InstructionConstants.OP_ANEWARRAY, 1),
        new SimpleInstruction(InstructionConstants.OP_DUP),
        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER0_CLASS_INDEX),
        new SimpleInstruction(InstructionConstants.OP_AASTORE),
        new SimpleInstruction(InstructionConstants.OP_DUP),
        new SimpleInstruction(InstructionConstants.OP_ICONST_1),
        new ConstantInstruction(InstructionConstants.OP_LDC, PARAMETER1_CLASS_INDEX),
        new SimpleInstruction(InstructionConstants.OP_AASTORE),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 0),
    };

    // AtomicIntegerFieldUpdater.newUpdater(..., "someField").
    // AtomicLongFieldUpdater.newUpdater(..., "someField").
    // AtomicReferenceFieldUpdater.newUpdater(..., "someField").
    private final Instruction[] NEW_UPDATER_INSTRUCTIONS = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, MEMBER_NAME_INDEX),
        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, 0),
    };


    private final ClassPool      programClassPool;
    private final ClassPool      libraryClassPool;
    private final WarningPrinter notePrinter;
    private final StringMatcher  noteFieldExceptionMatcher;
    private final StringMatcher  noteMethodExceptionMatcher;


    private final InstructionSequenceMatcher constantGetFieldMatcher =
        new InstructionSequenceMatcher(GET_FIELD_CONSTANTS,
                                       CONSTANT_GET_FIELD_INSTRUCTIONS);

    private final InstructionSequenceMatcher constantGetDeclaredFieldMatcher =
        new InstructionSequenceMatcher(GET_DECLARED_FIELD_CONSTANTS,
                                       CONSTANT_GET_FIELD_INSTRUCTIONS);

//    private final InstructionSequenceMatcher constantGetConstructorMatcher0 =
//        new InstructionSequenceMatcher(GET_CONSTRUCTOR_CONSTANTS,
//                                       CONSTANT_GET_CONSTRUCTOR_INSTRUCTIONS0);
//
//    private final InstructionSequenceMatcher constantGetDeclaredConstructorMatcher0 =
//        new InstructionSequenceMatcher(GET_DECLARED_CONSTRUCTOR_CONSTANTS,
//                                       CONSTANT_GET_CONSTRUCTOR_INSTRUCTIONS0);
//
//    private final InstructionSequenceMatcher constantGetConstructorMatcher1 =
//        new InstructionSequenceMatcher(GET_CONSTRUCTOR_CONSTANTS,
//                                       CONSTANT_GET_CONSTRUCTOR_INSTRUCTIONS1);
//
//    private final InstructionSequenceMatcher constantGetDeclaredConstructorMatcher1 =
//        new InstructionSequenceMatcher(GET_DECLARED_CONSTRUCTOR_CONSTANTS,
//                                       CONSTANT_GET_CONSTRUCTOR_INSTRUCTIONS1);
//
//    private final InstructionSequenceMatcher constantGetConstructorMatcher2 =
//        new InstructionSequenceMatcher(GET_CONSTRUCTOR_CONSTANTS,
//                                       CONSTANT_GET_CONSTRUCTOR_INSTRUCTIONS2);
//
//    private final InstructionSequenceMatcher constantGetDeclaredConstructorMatcher2 =
//        new InstructionSequenceMatcher(GET_DECLARED_CONSTRUCTOR_CONSTANTS,
//                                       CONSTANT_GET_CONSTRUCTOR_INSTRUCTIONS2);

    private final InstructionSequenceMatcher constantGetMethodMatcher0 =
        new InstructionSequenceMatcher(GET_METHOD_CONSTANTS,
                                       CONSTANT_GET_METHOD_INSTRUCTIONS0);

    private final InstructionSequenceMatcher constantGetDeclaredMethodMatcher0 =
        new InstructionSequenceMatcher(GET_DECLARED_METHOD_CONSTANTS,
                                       CONSTANT_GET_METHOD_INSTRUCTIONS0);

    private final InstructionSequenceMatcher constantGetMethodMatcher1 =
        new InstructionSequenceMatcher(GET_METHOD_CONSTANTS,
                                       CONSTANT_GET_METHOD_INSTRUCTIONS1);

    private final InstructionSequenceMatcher constantGetDeclaredMethodMatcher1 =
        new InstructionSequenceMatcher(GET_DECLARED_METHOD_CONSTANTS,
                                       CONSTANT_GET_METHOD_INSTRUCTIONS1);

    private final InstructionSequenceMatcher constantGetMethodMatcher2 =
        new InstructionSequenceMatcher(GET_METHOD_CONSTANTS,
                                       CONSTANT_GET_METHOD_INSTRUCTIONS2);

    private final InstructionSequenceMatcher constantGetDeclaredMethodMatcher2 =
        new InstructionSequenceMatcher(GET_DECLARED_METHOD_CONSTANTS,
                                       CONSTANT_GET_METHOD_INSTRUCTIONS2);

    private final InstructionSequenceMatcher constantGetIntegerUpdaterMatcher =
        new InstructionSequenceMatcher(NEW_INTEGER_UPDATER_CONSTANTS,
                                       CONSTANT_NEW_PRIMITIVE_UPDATER_INSTRUCTIONS);

    private final InstructionSequenceMatcher constantGetLongUpdaterMatcher =
        new InstructionSequenceMatcher(NEW_LONG_UPDATER_CONSTANTS,
                                       CONSTANT_NEW_PRIMITIVE_UPDATER_INSTRUCTIONS);

    private final InstructionSequenceMatcher constantGetReferenceUpdaterMatcher =
        new InstructionSequenceMatcher(NEW_REFERENCE_UPDATER_CONSTANTS,
                                       CONSTANT_NEW_REFERENCE_UPDATER_INSTRUCTIONS);

    private final InstructionSequenceMatcher getFieldMatcher =
        new InstructionSequenceMatcher(GET_FIELD_CONSTANTS,
                                       GET_FIELD_INSTRUCTIONS);

    private final InstructionSequenceMatcher getDeclaredFieldMatcher =
        new InstructionSequenceMatcher(GET_DECLARED_FIELD_CONSTANTS,
                                       GET_FIELD_INSTRUCTIONS);

//    private final InstructionSequenceMatcher getConstructorMatcher0 =
//        new InstructionSequenceMatcher(GET_CONSTRUCTOR_CONSTANTS,
//                                       GET_CONSTRUCTOR_INSTRUCTIONS0);
//
//    private final InstructionSequenceMatcher getDeclaredConstructorMatcher0 =
//        new InstructionSequenceMatcher(GET_DECLARED_CONSTRUCTOR_CONSTANTS,
//                                       GET_CONSTRUCTOR_INSTRUCTIONS0);

    private final InstructionSequenceMatcher getConstructorMatcher1 =
        new InstructionSequenceMatcher(GET_CONSTRUCTOR_CONSTANTS,
                                       GET_CONSTRUCTOR_INSTRUCTIONS1);

    private final InstructionSequenceMatcher getDeclaredConstructorMatcher1 =
        new InstructionSequenceMatcher(GET_DECLARED_CONSTRUCTOR_CONSTANTS,
                                       GET_CONSTRUCTOR_INSTRUCTIONS1);

    private final InstructionSequenceMatcher getConstructorMatcher2 =
        new InstructionSequenceMatcher(GET_CONSTRUCTOR_CONSTANTS,
                                       GET_CONSTRUCTOR_INSTRUCTIONS2);

    private final InstructionSequenceMatcher getDeclaredConstructorMatcher2 =
        new InstructionSequenceMatcher(GET_DECLARED_CONSTRUCTOR_CONSTANTS,
                                       GET_CONSTRUCTOR_INSTRUCTIONS2);

    private final InstructionSequenceMatcher getMethodMatcher0 =
        new InstructionSequenceMatcher(GET_METHOD_CONSTANTS,
                                       GET_METHOD_INSTRUCTIONS0);

    private final InstructionSequenceMatcher getDeclaredMethodMatcher0 =
        new InstructionSequenceMatcher(GET_DECLARED_METHOD_CONSTANTS,
                                       GET_METHOD_INSTRUCTIONS0);

    private final InstructionSequenceMatcher getMethodMatcher1 =
        new InstructionSequenceMatcher(GET_METHOD_CONSTANTS,
                                       GET_METHOD_INSTRUCTIONS1);

    private final InstructionSequenceMatcher getDeclaredMethodMatcher1 =
        new InstructionSequenceMatcher(GET_DECLARED_METHOD_CONSTANTS,
                                       GET_METHOD_INSTRUCTIONS1);

    private final InstructionSequenceMatcher getMethodMatcher2 =
        new InstructionSequenceMatcher(GET_METHOD_CONSTANTS,
                                       GET_METHOD_INSTRUCTIONS2);

    private final InstructionSequenceMatcher getDeclaredMethodMatcher2 =
        new InstructionSequenceMatcher(GET_DECLARED_METHOD_CONSTANTS,
                                       GET_METHOD_INSTRUCTIONS2);

    private final InstructionSequenceMatcher getIntegerUpdaterMatcher =
        new InstructionSequenceMatcher(NEW_INTEGER_UPDATER_CONSTANTS,
                                       NEW_UPDATER_INSTRUCTIONS);

    private final InstructionSequenceMatcher getLongUpdaterMatcher =
        new InstructionSequenceMatcher(NEW_LONG_UPDATER_CONSTANTS,
                                       NEW_UPDATER_INSTRUCTIONS);

    private final InstructionSequenceMatcher getReferenceUpdaterMatcher =
        new InstructionSequenceMatcher(NEW_REFERENCE_UPDATER_CONSTANTS,
                                       NEW_UPDATER_INSTRUCTIONS);

    private final MemberFinder memberFinder = new MemberFinder();


    // Fields acting as parameters for the visitors.
    private Clazz   referencedClass;
    private String descriptor;
    private boolean isDeclared;
    private boolean isField;



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
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        // Try to match the SomeClass.class.getField("someField") construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetFieldMatcher,
                       getFieldMatcher, true, false, null, null);

        // Try to match the SomeClass.class.getDeclaredField("someField") construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetDeclaredFieldMatcher,
                       getDeclaredFieldMatcher, true, true, null, null);

//        // Try to match the SomeClass.class.getConstructor(new Class[]
//        // {}) construct.
//        matchGetMember(clazz, method, codeAttribute, offset, instruction,
//                       cnull, //onstantGetConstructorMatcher0,
//                       getConstructorMatcher0, false, false,
//                       ClassConstants.METHOD_NAME_INIT, null);
//
//        // Try to match the SomeClass.class.getDeclaredConstructor(new Class[]
//        // {}) construct.
//        matchGetMember(clazz, method, codeAttribute, offset, instruction,
//                       null, //constantGetDeclaredConstructorMatcher0,
//                       getDeclaredConstructorMatcher0, false, true,
//                       ClassConstants.METHOD_NAME_INIT, null);

        // Try to match the SomeClass.class.getConstructor(new Class[]
        // { A.class }) construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       null, //constantGetConstructorMatcher1,
                       getConstructorMatcher1, false, false,
                       ClassConstants.METHOD_NAME_INIT, null);

        // Try to match the SomeClass.class.getDeclaredConstructor(new Class[]
        // { A.class }) construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       null, //constantGetDeclaredConstructorMatcher1,
                       getDeclaredConstructorMatcher1, false, true,
                       ClassConstants.METHOD_NAME_INIT, null);

        // Try to match the SomeClass.class.getConstructor(new Class[]
        // { A.class, B.class }) construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       null, //constantGetConstructorMatcher2,
                       getConstructorMatcher2, false, false,
                       ClassConstants.METHOD_NAME_INIT, null);

        // Try to match the SomeClass.class.getDeclaredConstructor(new Class[]
        // { A.class, B.class }) construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       null, //constantGetDeclaredConstructorMatcher2,
                       getDeclaredConstructorMatcher2, false, true,
                       ClassConstants.METHOD_NAME_INIT, null);

        // Try to match the SomeClass.class.getMethod("someMethod", new Class[]
        // {}) construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetMethodMatcher0,
                       getMethodMatcher0, false, false, null, null);

        // Try to match the SomeClass.class.getDeclaredMethod("someMethod",
        // new Class[] {}) construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetDeclaredMethodMatcher0,
                       getDeclaredMethodMatcher0, false, true, null, null);

        // Try to match the SomeClass.class.getMethod("someMethod", new Class[]
        // { A.class }) construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetMethodMatcher1,
                       getMethodMatcher1, false, false, null, null);

        // Try to match the SomeClass.class.getDeclaredMethod("someMethod",
        //  new Class[] { A.class }) construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetDeclaredMethodMatcher1,
                       getDeclaredMethodMatcher1, false, true, null, null);

        // Try to match the SomeClass.class.getMethod("someMethod", new Class[]
        // { A.class, B.class }) construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetMethodMatcher2,
                       getMethodMatcher2, false, false, null, null);

        // Try to match the SomeClass.class.getDeclaredMethod("someMethod",
        // new Class[] { A.class, B.class }) construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetDeclaredMethodMatcher2,
                       getDeclaredMethodMatcher2, false, true, null, null);

        // Try to match the AtomicIntegerFieldUpdater.newUpdater(
        // SomeClass.class, "someField") construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetIntegerUpdaterMatcher,
                       getIntegerUpdaterMatcher, true, false, null,
                       "" + ClassConstants.TYPE_INT);

        // Try to match the AtomicLongFieldUpdater.newUpdater(
        // SomeClass.class, "someField") construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetLongUpdaterMatcher,
                       getLongUpdaterMatcher, true, false, null,
                       "" + ClassConstants.TYPE_LONG);

        // Try to match the AtomicReferenceFieldUpdater.newUpdater(
        // SomeClass.class, SomeClass.class, "someField") construct.
        matchGetMember(clazz, method, codeAttribute, offset, instruction,
                       constantGetReferenceUpdaterMatcher,
                       getReferenceUpdaterMatcher, true, false, null, null);
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
                                boolean                    isDeclared,
                                String                     defaultName,
                                String                     defaultDescriptor)
    {
        if (constantSequenceMatcher != null)
        {
            // Try to match the next instruction in the constant sequence.
            instruction.accept(clazz, method, codeAttribute, offset,
                               constantSequenceMatcher);

            // Did we find a match to fill out the string constant?
            if (constantSequenceMatcher.isMatching())
            {
                initializeStringReference(clazz,
                                          constantSequenceMatcher,
                                          isField,
                                          isDeclared,
                                          defaultDescriptor);

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
            // Print out a note about the dynamic invocation.
            printDynamicInvocationNote(clazz,
                                       variableSequenceMatcher,
                                       isField,
                                       isDeclared,
                                       defaultName,
                                       defaultDescriptor);
        }
    }


    /**
     * Initializes the reference of the matched string constant to the
     * referenced class member and its class.
     */
    private void initializeStringReference(Clazz                      clazz,
                                           InstructionSequenceMatcher constantSequenceMatcher,
                                           boolean                    isField,
                                           boolean                    isDeclared,
                                           String                     defaultDescriptor)
    {
        this.isField    = isField;
        this.isDeclared = isDeclared;

        // Get the member's class.
        int classIndex = constantSequenceMatcher.matchedConstantIndex(CLASS_INDEX);
        clazz.constantPoolEntryAccept(classIndex, this);

        // Get the field's reference type, if applicable.
        int typeClassIndex = constantSequenceMatcher.matchedConstantIndex(TYPE_CLASS_INDEX);
        descriptor = typeClassIndex <= 0 ? defaultDescriptor :
            ClassUtil.internalTypeFromClassName(clazz.getClassName(typeClassIndex));

        // Fill out the matched string constant.
        int memberNameIndex = constantSequenceMatcher.matchedConstantIndex(MEMBER_NAME_INDEX);
        clazz.constantPoolEntryAccept(memberNameIndex, this);
    }


    // Implementations for ConstantVisitor.

    /**
     * Remembers the referenced class.
     */
    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        if (DEBUG)
        {
            System.out.println("DynamicMemberReferenceInitializer: ["+clazz.getName()+"] matched class ["+classConstant.getName(clazz)+"]");
        }

        // Remember the referenced class.
        referencedClass = ClassUtil.isInternalArrayType(classConstant.getName(clazz)) ?
            null :
            classConstant.referencedClass;
    }


    /**
     * Fills out the link to the referenced class member.
     */
    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        if (referencedClass != null)
        {
            String name = stringConstant.getString(clazz);

            if (DEBUG)
            {
                System.out.println("DynamicMemberReferenceInitializer: ["+clazz.getName()+"] matched string ["+name+"]");
            }

            // See if we can find the referenced class member locally, or
            // somewhere in the hierarchy.
            Member referencedMember = isDeclared ? isField ?
                (Member)referencedClass.findField(name, descriptor) :
                (Member)referencedClass.findMethod(name, descriptor) :
                (Member)memberFinder.findMember(clazz,
                                                referencedClass,
                                                name,
                                                descriptor,
                                                isField);
            if (referencedMember != null)
            {
                stringConstant.referencedMember = referencedMember;
                stringConstant.referencedClass  = isDeclared ?
                    referencedClass :
                    memberFinder.correspondingClass();
            }
        }
    }


    // Small utility methods.

    /**
     * Prints out a note on the matched dynamic invocation, if necessary.
     */
    private void printDynamicInvocationNote(Clazz                      clazz,
                                            InstructionSequenceMatcher noteSequenceMatcher,
                                            boolean                    isField,
                                            boolean                    isDeclared,
                                            String                     defaultName,
                                            String                     defaultDescriptor)
    {
        // Print out a note about the dynamic invocation.
        if (notePrinter != null &&
            notePrinter.accepts(clazz.getName()))
        {
            // Is the class member name in the list of exceptions?
            StringMatcher noteExceptionMatcher = isField ?
                noteFieldExceptionMatcher :
                noteMethodExceptionMatcher;

            int memberNameIndex = noteSequenceMatcher.matchedConstantIndex(MEMBER_NAME_INDEX);
            String memberName = memberNameIndex <= 0 ? defaultName :
                clazz.getStringString(memberNameIndex);

            if (noteExceptionMatcher == null ||
                !noteExceptionMatcher.matches(memberName))
            {
                // Compose the external member name and partial descriptor.
                String externalMemberDescription = memberName;

                if (!isField)
                {
                    externalMemberDescription += '(';
                    for (int count = 0; count < 2; count++)
                    {
                        int memberArgumentIndex = noteSequenceMatcher.matchedConstantIndex(
                            PARAMETER0_CLASS_INDEX + count);
                        if (memberArgumentIndex > 0)
                        {
                            if (count > 0)
                            {
                                externalMemberDescription += ',';
                            }
                            String className = clazz.getClassName(memberArgumentIndex);
                            externalMemberDescription += ClassUtil.isInternalArrayType(className) ?
                                ClassUtil.externalType(className) :
                                ClassUtil.externalClassName(className);
                        }
                    }
                    externalMemberDescription += ')';
                }

                // Print out the actual note.
                notePrinter.print(clazz.getName(),
                                  "Note: " +
                                  ClassUtil.externalClassName(clazz.getName()) +
                                  " accesses a " +
                                  (isDeclared ? "declared " : "") +
                                  (isField    ? "field" :
                                   memberName.equals(ClassConstants.METHOD_NAME_INIT) ?
                                                "constructor" : "method") +
                                  " '" +
                                  externalMemberDescription +
                                  "' dynamically");

                // Print out notes about potential candidates.
                ClassVisitor classVisitor;

                if (isField)
                {
                    classVisitor = defaultDescriptor == null ?
                       new AllFieldVisitor(
                       new MemberNameFilter(memberName, this)) :
                       new AllFieldVisitor(
                       new MemberNameFilter(memberName,
                       new MemberDescriptorFilter(defaultDescriptor, this)));
                }
                else
                {
                    // Compose the partial method descriptor.
                    String methodDescriptor = "(";
                    for (int count = 0; count < 2; count++)
                    {
                        int memberArgumentIndex = noteSequenceMatcher.matchedConstantIndex(PARAMETER0_CLASS_INDEX + count);
                        if (memberArgumentIndex > 0)
                        {
                            String className = clazz.getClassName(memberArgumentIndex);
                            methodDescriptor += ClassUtil.isInternalArrayType(className) ?
                                className :
                                ClassUtil.internalTypeFromClassName(className);
                        }
                    }
                    methodDescriptor += ")L***;";

                    classVisitor =
                        new AllMethodVisitor(
                        new MemberNameFilter(memberName,
                        new MemberDescriptorFilter(methodDescriptor, this)));
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
}