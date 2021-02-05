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
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.util.StringMatcher;

/**
 * This InstructionVisitor initializes any constant <code>Class.forName</code> or
 * <code>.class</code> references of all classes it visits. More specifically,
 * it fills out the references of string constant pool entries that refer to a
 * class in the program class pool or in the library class pool.
 * <p>
 * It optionally prints notes if on usage of
 * <code>(SomeClass)Class.forName(variable).newInstance()</code>.
 * <p>
 * The class hierarchy must be initialized before using this visitor.
 *
 * @see ClassSuperHierarchyInitializer
 *
 * @author Eric Lafortune
 */
public class DynamicClassReferenceInitializer
extends      SimplifiedVisitor
implements   InstructionVisitor,
             ConstantVisitor,
             AttributeVisitor
{
    public static final int X = InstructionSequenceMatcher.X;
    public static final int Y = InstructionSequenceMatcher.Y;
    public static final int Z = InstructionSequenceMatcher.Z;

    public static final int A = InstructionSequenceMatcher.A;
    public static final int B = InstructionSequenceMatcher.B;
    public static final int C = InstructionSequenceMatcher.C;
    public static final int D = InstructionSequenceMatcher.D;


    private final Constant[] CLASS_FOR_NAME_CONSTANTS = new Constant[]
    {
        // 0
        new MethodrefConstant(1, 2, null, null),
        new ClassConstant(3, null),
        new NameAndTypeConstant(4, 5),
        new Utf8Constant(ClassConstants.NAME_JAVA_LANG_CLASS),
        new Utf8Constant(ClassConstants.METHOD_NAME_CLASS_FOR_NAME),
        new Utf8Constant(ClassConstants.METHOD_TYPE_CLASS_FOR_NAME),

        // 6
        new MethodrefConstant(1, 7, null, null),
        new NameAndTypeConstant(8, 9),
        new Utf8Constant(ClassConstants.METHOD_NAME_NEW_INSTANCE),
        new Utf8Constant(ClassConstants.METHOD_TYPE_NEW_INSTANCE),

        // 10
        new MethodrefConstant(1, 11, null, null),
        new NameAndTypeConstant(12, 13),
        new Utf8Constant(ClassConstants.METHOD_NAME_CLASS_GET_COMPONENT_TYPE),
        new Utf8Constant(ClassConstants.METHOD_TYPE_CLASS_GET_COMPONENT_TYPE),
    };

    // Class.forName("SomeClass").
    private final Instruction[] CONSTANT_CLASS_FOR_NAME_INSTRUCTIONS = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, X),
        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, 0),
    };

    // (SomeClass)Class.forName(someName).newInstance().
    private final Instruction[] CLASS_FOR_NAME_CAST_INSTRUCTIONS = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, 0),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 6),
        new ConstantInstruction(InstructionConstants.OP_CHECKCAST, X),
    };


//    private Constant[] DOT_CLASS_JAVAC_CONSTANTS = new Constant[]
//    {
//        new MethodrefConstant(A, 1, null, null),
//        new NameAndTypeConstant(2, 3),
//        new Utf8Constant(ClassConstants.METHOD_NAME_DOT_CLASS_JAVAC),
//        new Utf8Constant(ClassConstants.METHOD_TYPE_DOT_CLASS_JAVAC),
//    };

    private final Constant[] DOT_CLASS_JAVAC_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(A, 1, null, null),
        new NameAndTypeConstant(B, 2),
        new Utf8Constant(ClassConstants.METHOD_TYPE_DOT_CLASS_JAVAC),
    };

    // SomeClass.class = class$("SomeClass") (javac).
    private final Instruction[] DOT_CLASS_JAVAC_INSTRUCTIONS = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, X),
        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, 0),
    };


//    private Constant[] DOT_CLASS_JIKES_CONSTANTS = new Constant[]
//    {
//        new MethodrefConstant(A, 1, null, null),
//        new NameAndTypeConstant(2, 3),
//        new Utf8Constant(ClassConstants.METHOD_NAME_DOT_CLASS_JIKES),
//        new Utf8Constant(ClassConstants.METHOD_TYPE_DOT_CLASS_JIKES),
//    };

    private final Constant[] DOT_CLASS_JIKES_CONSTANTS = new Constant[]
    {
        new MethodrefConstant(A, 1, null, null),
        new NameAndTypeConstant(B, 2),
        new Utf8Constant(ClassConstants.METHOD_TYPE_DOT_CLASS_JIKES),
    };

    // SomeClass.class = class("SomeClass", false) (jikes).
    private final Instruction[] DOT_CLASS_JIKES_INSTRUCTIONS = new Instruction[]
    {
        new ConstantInstruction(InstructionConstants.OP_LDC, X),
        new SimpleInstruction(InstructionConstants.OP_ICONST_0),
        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, 0),
    };

    // return Class.forName(v0).
    private final Instruction[] DOT_CLASS_JAVAC_IMPLEMENTATION_INSTRUCTIONS = new Instruction[]
    {
        new VariableInstruction(InstructionConstants.OP_ALOAD_0),
        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, 0),
        new SimpleInstruction(InstructionConstants.OP_ARETURN),
    };

    // return Class.forName(v0), if (!v1) .getComponentType().
    private final Instruction[] DOT_CLASS_JIKES_IMPLEMENTATION_INSTRUCTIONS = new Instruction[]
    {
        new VariableInstruction(InstructionConstants.OP_ALOAD_0),
        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, 0),
        new VariableInstruction(InstructionConstants.OP_ALOAD_1),
        new BranchInstruction(InstructionConstants.OP_IFNE, +6),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 10),
        new SimpleInstruction(InstructionConstants.OP_ARETURN),
    };

    // return Class.forName(v0).getComponentType().
    private final Instruction[] DOT_CLASS_JIKES_IMPLEMENTATION_INSTRUCTIONS2 = new Instruction[]
    {
        new VariableInstruction(InstructionConstants.OP_ALOAD_0),
        new ConstantInstruction(InstructionConstants.OP_INVOKESTATIC, 0),
        new ConstantInstruction(InstructionConstants.OP_INVOKEVIRTUAL, 10),
        new SimpleInstruction(InstructionConstants.OP_ARETURN),
    };


    private final ClassPool      programClassPool;
    private final ClassPool      libraryClassPool;
    private final WarningPrinter missingNotePrinter;
    private final WarningPrinter dependencyWarningPrinter;
    private final WarningPrinter notePrinter;
    private final StringMatcher  noteExceptionMatcher;


    private final InstructionSequenceMatcher constantClassForNameMatcher =
        new InstructionSequenceMatcher(CLASS_FOR_NAME_CONSTANTS,
                                       CONSTANT_CLASS_FOR_NAME_INSTRUCTIONS);

    private final InstructionSequenceMatcher classForNameCastMatcher =
        new InstructionSequenceMatcher(CLASS_FOR_NAME_CONSTANTS,
                                       CLASS_FOR_NAME_CAST_INSTRUCTIONS);

    private final InstructionSequenceMatcher dotClassJavacMatcher =
        new InstructionSequenceMatcher(DOT_CLASS_JAVAC_CONSTANTS,
                                       DOT_CLASS_JAVAC_INSTRUCTIONS);

    private final InstructionSequenceMatcher dotClassJikesMatcher =
        new InstructionSequenceMatcher(DOT_CLASS_JIKES_CONSTANTS,
                                       DOT_CLASS_JIKES_INSTRUCTIONS);

    private final InstructionSequenceMatcher dotClassJavacImplementationMatcher =
        new InstructionSequenceMatcher(CLASS_FOR_NAME_CONSTANTS,
                                       DOT_CLASS_JAVAC_IMPLEMENTATION_INSTRUCTIONS);

    private final InstructionSequenceMatcher dotClassJikesImplementationMatcher =
        new InstructionSequenceMatcher(CLASS_FOR_NAME_CONSTANTS,
                                       DOT_CLASS_JIKES_IMPLEMENTATION_INSTRUCTIONS);

    private final InstructionSequenceMatcher dotClassJikesImplementationMatcher2 =
        new InstructionSequenceMatcher(CLASS_FOR_NAME_CONSTANTS,
                                       DOT_CLASS_JIKES_IMPLEMENTATION_INSTRUCTIONS2);


    // A field acting as a return variable for the visitors.
    private boolean isClassForNameInvocation;


    /**
     * Creates a new DynamicClassReferenceInitializer that optionally prints
     * warnings and notes, with optional class specifications for which never
     * to print notes.
     */
    public DynamicClassReferenceInitializer(ClassPool      programClassPool,
                                            ClassPool      libraryClassPool,
                                            WarningPrinter missingNotePrinter,
                                            WarningPrinter dependencyWarningPrinter,
                                            WarningPrinter notePrinter,
                                            StringMatcher  noteExceptionMatcher)
    {
        this.programClassPool         = programClassPool;
        this.libraryClassPool         = libraryClassPool;
        this.missingNotePrinter       = missingNotePrinter;
        this.dependencyWarningPrinter = dependencyWarningPrinter;
        this.notePrinter              = notePrinter;
        this.noteExceptionMatcher     = noteExceptionMatcher;
    }


    // Implementations for InstructionVisitor.

    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction)
    {
        // Try to match the (SomeClass)Class.forName(someName).newInstance()
        // construct. Apply this matcher first, so the next matcher can still
        // reset it after the first instruction.
        instruction.accept(clazz, method, codeAttribute, offset,
                           classForNameCastMatcher);

        // Did we find a match?
        if (classForNameCastMatcher.isMatching())
        {
            // Print out a note about the construct.
            clazz.constantPoolEntryAccept(classForNameCastMatcher.matchedConstantIndex(X), this);
        }

        // Try to match the Class.forName("SomeClass") construct.
        instruction.accept(clazz, method, codeAttribute, offset,
                           constantClassForNameMatcher);

        // Did we find a match?
        if (constantClassForNameMatcher.isMatching())
        {
            // Fill out the matched string constant.
            clazz.constantPoolEntryAccept(constantClassForNameMatcher.matchedConstantIndex(X), this);

            // Don't look for the dynamic construct.
            classForNameCastMatcher.reset();
        }

        // Try to match the javac .class construct.
        instruction.accept(clazz, method, codeAttribute, offset,
                           dotClassJavacMatcher);

        // Did we find a match?
        if (dotClassJavacMatcher.isMatching() &&
            isDotClassMethodref(clazz, dotClassJavacMatcher.matchedConstantIndex(0)))
        {
            // Fill out the matched string constant.
            clazz.constantPoolEntryAccept(dotClassJavacMatcher.matchedConstantIndex(X), this);
        }

        // Try to match the jikes .class construct.
        instruction.accept(clazz, method, codeAttribute, offset,
                           dotClassJikesMatcher);

        // Did we find a match?
        if (dotClassJikesMatcher.isMatching() &&
            isDotClassMethodref(clazz, dotClassJikesMatcher.matchedConstantIndex(0)))
        {
            // Fill out the matched string constant.
            clazz.constantPoolEntryAccept(dotClassJikesMatcher.matchedConstantIndex(X), this);
        }
    }


    // Implementations for ConstantVisitor.

    /**
     * Fills out the link to the referenced class.
     */
    public void visitStringConstant(Clazz clazz, StringConstant stringConstant)
    {
        // Save a reference to the corresponding class.
        String externalClassName = stringConstant.getString(clazz);
        String internalClassName = ClassUtil.internalClassName(
                                   ClassUtil.externalBaseType(externalClassName));

        stringConstant.referencedClass = findClass(clazz.getName(), internalClassName);
    }


    /**
     * Prints out a note about the class cast to this class, if applicable.
     */
    public void visitClassConstant(Clazz clazz, ClassConstant classConstant)
    {
        // Print out a note about the class cast.
        if (noteExceptionMatcher == null ||
            !noteExceptionMatcher.matches(classConstant.getName(clazz)))
        {
            notePrinter.print(clazz.getName(),
                              classConstant.getName(clazz),
                              "Note: " +
                              ClassUtil.externalClassName(clazz.getName()) +
                              " calls '(" +
                              ClassUtil.externalClassName(classConstant.getName(clazz)) +
                              ")Class.forName(variable).newInstance()'");
        }
    }


    /**
     * Checks whether the referenced method is a .class method.
     */
    public void visitMethodrefConstant(Clazz clazz, MethodrefConstant methodrefConstant)
    {
        String methodType = methodrefConstant.getType(clazz);

        // Do the method's class and type match?
        if (methodType.equals(ClassConstants.METHOD_TYPE_DOT_CLASS_JAVAC) ||
            methodType.equals(ClassConstants.METHOD_TYPE_DOT_CLASS_JIKES))
        {
            String methodName = methodrefConstant.getName(clazz);

            // Does the method's name match one of the special names?
            isClassForNameInvocation =
                methodName.equals(ClassConstants.METHOD_NAME_DOT_CLASS_JAVAC) ||
                methodName.equals(ClassConstants.METHOD_NAME_DOT_CLASS_JIKES);

            if (isClassForNameInvocation)
            {
                return;
            }

            String className = methodrefConstant.getClassName(clazz);

            // Note that we look for the class by name, since the referenced
            // class has not been initialized yet.
            Clazz referencedClass = programClassPool.getClass(className);
            if (referencedClass != null)
            {
                // Check if the code of the referenced method is .class code.
                // Note that we look for the method by name and type, since the
                // referenced method has not been initialized yet.
                referencedClass.methodAccept(methodName,
                                             methodType,
                                             new AllAttributeVisitor(this));
            }
        }
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        // Check whether this is class$(String), as generated by javac, or
        // class(String, boolean), as generated by jikes, or an optimized
        // version.
        isClassForNameInvocation =
            isDotClassMethodCode(clazz, method, codeAttribute,
                                 dotClassJavacImplementationMatcher, 5)  ||
            isDotClassMethodCode(clazz, method, codeAttribute,
                                 dotClassJikesImplementationMatcher, 12) ||
            isDotClassMethodCode(clazz, method, codeAttribute,
                                 dotClassJikesImplementationMatcher2, 8);
    }


    // Small utility methods.

    /**
     * Returns whether the given method reference corresponds to a .class
     * method, as generated by javac or by jikes.
     */
    private boolean isDotClassMethodref(Clazz clazz, int methodrefConstantIndex)
    {
        isClassForNameInvocation = false;

        // Check if the code of the referenced method is .class code.
        clazz.constantPoolEntryAccept(methodrefConstantIndex, this);

        return isClassForNameInvocation;
    }


    /**
     * Returns whether the first whether the first instructions of the
     * given code attribute match with the given instruction matcher.
     */
    private boolean isDotClassMethodCode(Clazz                      clazz,
                                         Method                     method,
                                         CodeAttribute              codeAttribute,
                                         InstructionSequenceMatcher codeMatcher,
                                         int                        codeLength)
    {
        // Check the minimum code length.
        if (codeAttribute.u4codeLength < codeLength)
        {
            return false;
        }

        // Check the actual instructions.
        codeMatcher.reset();
        codeAttribute.instructionsAccept(clazz, method, 0, codeLength, codeMatcher);
        return codeMatcher.isMatching();
    }


    /**
     * Returns the class with the given name, either for the program class pool
     * or from the library class pool, or <code>null</code> if it can't be found.
     */
    private Clazz findClass(String referencingClassName, String name)
    {
        // Is it an array type?
        if (ClassUtil.isInternalArrayType(name))
        {
            // Ignore any primitive array types.
            if (!ClassUtil.isInternalClassType(name))
            {
                return null;
            }

            // Strip the array part.
            name = ClassUtil.internalClassNameFromClassType(name);
        }

        // First look for the class in the program class pool.
        Clazz clazz = programClassPool.getClass(name);

        // Otherwise look for the class in the library class pool.
        if (clazz == null)
        {
            clazz = libraryClassPool.getClass(name);

            if (clazz == null &&
                missingNotePrinter != null)
            {
                // We didn't find the superclass or interface. Print a note.
                missingNotePrinter.print(referencingClassName,
                                         name,
                                         "Note: " +
                                         ClassUtil.externalClassName(referencingClassName) +
                                         ": can't find dynamically referenced class " +
                                         ClassUtil.externalClassName(name));
            }
        }
        else if (dependencyWarningPrinter != null)
        {
            // The superclass or interface was found in the program class pool.
            // Print a warning.
            dependencyWarningPrinter.print(referencingClassName,
                                           name,
                                           "Warning: library class " +
                                           ClassUtil.externalClassName(referencingClassName) +
                                           " depends dynamically on program class " +
                                           ClassUtil.externalClassName(name));
        }

        return clazz;
    }
}
