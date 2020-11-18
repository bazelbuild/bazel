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
package proguard.classfile.editor;

import proguard.classfile.*;
import proguard.classfile.attribute.*;
import proguard.classfile.attribute.visitor.*;
import proguard.classfile.constant.Constant;
import proguard.classfile.instruction.*;
import proguard.classfile.util.*;
import proguard.classfile.visitor.*;
import proguard.obfuscate.*;

import java.util.*;

import static proguard.classfile.ClassConstants.*;

/**
 * This editor allows to build and/or edit classes (ProgramClass instances).
 * It provides methods to easily add fields and methods to classes.
 *
 * @author Johan Leys
 */
public class SimplifiedClassEditor
extends      SimplifiedVisitor
implements
             // Implementation interfaces.
             AttributeVisitor
{
    private static final String EXTRA_INIT_METHOD_NAME       = "init$";
    private static final String EXTRA_INIT_METHOD_DESCRIPTOR = "()V";

    private final ProgramClass       programClass;
    private final ClassEditor        classEditor;
    private final ConstantPoolEditor constantPoolEditor;
    private final NameFactory        nameFactory;

    private String superClassName;

    private final List<CodeComposer> methodComposers = new ArrayList<CodeComposer>();

    private Instruction[] instructions;

    /**
     * Creates a new SimplifiedClassEditor for the Java class with the given
     * name.
     *
     * @param u2accessFlags access flags for the new class.
     * @param className     the fully qualified name of the new class.
     *
     * @see ClassConstants
     */
    public SimplifiedClassEditor(int u2accessFlags, String className)
    {
        this(u2accessFlags, className, null);
    }


    /**
     * Creates a new SimplifiedClassEditor for the Java class with the given
     * name and super class.
     *
     * @param u2accessFlags  access flags for the new class.
     * @param className      the fully qualified name of the new class.
     * @param superclassName the fully qualified name of the super class.
     *
     * @see ClassConstants
     */
    public SimplifiedClassEditor(int    u2accessFlags,
                                 String className,
                                 String superclassName)
    {
        this(new ProgramClass(ClassConstants.CLASS_VERSION_1_2,
                              1,
                              new Constant[10],
                              u2accessFlags,
                              0,
                              0));

        programClass.u2thisClass =
            constantPoolEditor.addClassConstant(className, programClass);

        if (superclassName != null)
        {
            programClass.u2superClass =
                constantPoolEditor.addClassConstant(superclassName, null);
            this.superClassName = superclassName;
        }
    }


    /**
     * Creates a new SimplifiedClassEditor for the given class.
     *
     * @param programClass the class to be edited.
     */
    public SimplifiedClassEditor(ProgramClass programClass)
    {
        this.programClass  = programClass;
        classEditor        = new ClassEditor(programClass);
        constantPoolEditor = new ConstantPoolEditor(programClass);
        nameFactory        = UniqueMemberNameFactory.newInjectedMemberNameFactory(programClass);
    }


    /**
     * Finalizes the editing of the class. This method does not initialize
     * references to/from related classes.
     * At least one of the finishEditing methods should be called before
     * calling {@link #getProgramClass}.
     *
     * @see #finishEditing(ClassPool, ClassPool)
     */
    public void finishEditing() {
        for (CodeComposer composer : methodComposers) {
            composer.finishEditing();
        }
    }

    /**
     * Finalizes the editing of the class, and initializes all references
     * of the edited class w.r.t. the given program and library class pool.
     * At least one of the finishEditing methods should be called before
     * calling {@link #getProgramClass}.
     *
     * @param programClassPool the program class pool
     * @param libraryClassPool the library class pool
     */
    public void finishEditing(ClassPool programClassPool,
                              ClassPool libraryClassPool) {
        for (CodeComposer composer : methodComposers) {
            composer.finishEditing();
        }

        // Initialize all references to/from the edited class.
        if (superClassName != null)
        {
            new ClassSuperHierarchyInitializer(programClassPool, libraryClassPool, null, null).visitProgramClass(programClass);
            new ClassSubHierarchyInitializer().visitProgramClass(programClass);
        }
        new ClassReferenceInitializer(programClassPool, libraryClassPool).visitProgramClass(programClass);
    }

    /**
     * Returns the edited ProgramClass instance.
     * Make sure to call one of the finishEditing methods after finishing editing,
     * before calling this method.
     *
     * @return the edited ProgramClass instance.
     *
     * @see #finishEditing()
     * @see #finishEditing(ClassPool, ClassPool)
     */
    public ProgramClass getProgramClass()
    {
        return programClass;
    }


    /**
     * Adds the given class constant to the edited class.
     *
     * @param name            the class name to be added.
     * @param referencedClass the corresponding referenced class.
     *
     * @return the constant pool index of the ClassConstant.
     */
    public int addClassConstant(String name,
                                Clazz  referencedClass) {
        return constantPoolEditor.addClassConstant(name, referencedClass);
    }

    /**
     * Adds a new field to the edited class.
     *
     * @param u2accessFlags    acces flags for the new field.
     * @param fieldName        name of the new field.
     * @param fieldDescriptor  descriptor of the new field.
     *
     * @return this SimpleClassEditor.
     */
    public SimplifiedClassEditor addField(int    u2accessFlags,
                                          String fieldName,
                                          String fieldDescriptor)
    {
        Field field = new ProgramField(u2accessFlags,
                                       constantPoolEditor.addUtf8Constant(fieldName),
                                       constantPoolEditor.addUtf8Constant(fieldDescriptor),
                                       null);
        classEditor.addField(field);
        return this;
    }


    /**
     * Adds a new method to the edited class. The returned composer can be used
     * to attach code to the method.
     *
     * @param u2accessFlags         acces flags for the new method.
     * @param methodName            name of the new method.
     * @param methodDescriptor      descriptor of the new method.
     * @param maxCodeFragmentLength maximum length for the code fragment of the
     *                              new method.
     *
     * @return the composer for adding code to the created method.
     */
    public CompactCodeAttributeComposer addMethod(int    u2accessFlags,
                                                  String methodName,
                                                  String methodDescriptor,
                                                  int    maxCodeFragmentLength)
    {
        return addMethod(u2accessFlags, methodName, methodDescriptor, null, maxCodeFragmentLength);
    }

    /**
     * Adds a new method to the edited class. The returned composer can be used
     * to attach code to the method.
     *
     * @param u2accessFlags         acces flags for the new method.
     * @param methodName            name of the new method.
     * @param methodDescriptor      descriptor of the new method.
     * @param referencedClasses     the classes referenced by the method descriptor.
     * @param maxCodeFragmentLength maximum length for the code fragment of the
     *                              new method.
     *
     * @return the composer for adding code to the created method.
     */
    public CompactCodeAttributeComposer addMethod(int     u2accessFlags,
                                                  String  methodName,
                                                  String  methodDescriptor,
                                                  Clazz[] referencedClasses,
                                                  int     maxCodeFragmentLength)
    {
        ProgramMethod method = new ProgramMethod(u2accessFlags,
            constantPoolEditor.addUtf8Constant(methodName),
            constantPoolEditor.addUtf8Constant(methodDescriptor),
                                               referencedClasses);
        CodeComposer composer = new CodeComposer(method, maxCodeFragmentLength);
        methodComposers.add(composer);
        return composer;
    }


    /**
     * Adds a new method to the edited class, with the given instructions array.
     *
     * @param u2accessFlags         acces flags for the new method.
     * @param methodName            name of the new method.
     * @param methodDescriptor      descriptor of the new method.
     * @param instructions          the instructions of the new method.
     */
    public ProgramMethod addMethod(int           u2accessFlags,
                                   String        methodName,
                                   String        methodDescriptor,
                                   Instruction[] instructions)
    {
        return addMethod(u2accessFlags, methodName, methodDescriptor, instructions, null, null);
    }


    /**
     * Adds the given static initializer instructions to the edited class.
     * If the class already contains a static initializer, the new instructions
     * will be appended to the existing initializer.
     *
     * @param instructions                 the instructions to be added.
     * @param mergeIntoExistingInitializer indicates whether the instructions should
     *                                     be added to the existing static initializer
     *                                     (if it exists), or if a new method should
     *                                     be created, which is then called from the
     *                                     existing initializer.
     */
    public void addStaticInitializerInstructions(Instruction[] instructions,
                                                 boolean       mergeIntoExistingInitializer)
    {
        Method method = programClass.findMethod(METHOD_NAME_CLINIT, METHOD_TYPE_CLINIT);

        if (method == null) {
            addMethod(ACC_STATIC, METHOD_NAME_CLINIT, METHOD_TYPE_CLINIT,
                      instructions, null,
                      new SimpleInstruction(InstructionConstants.OP_RETURN));
        }
        else {
            if (!mergeIntoExistingInitializer)
            {
                // Create a new static initializer.
                ProgramMethod newMethod =
                    addMethod(ACC_STATIC,
                              nameFactory.nextName(), "()V",
                              instructions,
                              null,
                              new SimpleInstruction(InstructionConstants.OP_RETURN));

                // Call the new initializer from the existing one.
                InstructionSequenceBuilder builder = new InstructionSequenceBuilder(programClass);
                builder.invokestatic(programClass.getName(),
                                      newMethod.getName(programClass),
                                      "()V",
                                      programClass,
                                      newMethod);
                instructions = builder.instructions();
            }
            CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor();
            ((ProgramMethod) method).attributesAccept(programClass,
                                                      new CodeAttributeEditorResetter(codeAttributeEditor));
            codeAttributeEditor.insertBeforeOffset(0, instructions);
            ((ProgramMethod) method).attributesAccept(programClass,
                                                      codeAttributeEditor);
        }
    }

    /**
     * Adds the given initialization instructions to the edited class.
     *
     * - If the class doesn't contain a constructor yet, it will be created,
     *   and the instructions will be added to this constructor.
     * - If there is a single super-calling constructor, the instructions will
     *   be added at the beginning of it's code attribute.
     * - If there are multiple super-calling constructors, a new private
     *   parameterless helper method will be created, to which the instructions
     *   will be added. An invocation to this new method will be added at the
     *   beginning of the code attribute of all super-calling constructors.
     *
     * @param instructions the instructions to be added.
     */
    public void addInitializerInstructions(Instruction[] instructions)
    {
        Method method = programClass.findMethod(METHOD_NAME_INIT, null);

        if (method == null) {
            // First call the super constructor.
            Instruction[] firstInstruction = {
                new VariableInstruction(InstructionConstants.OP_ALOAD_0),
                new ConstantInstruction(
                    InstructionConstants.OP_INVOKESPECIAL,
                    constantPoolEditor.addMethodrefConstant(programClass.getSuperClass().getName(), METHOD_NAME_INIT, METHOD_TYPE_INIT, null, null))
            };

            // End by calling return.
            SimpleInstruction lastInstruction =
                new SimpleInstruction(InstructionConstants.OP_RETURN);

            addMethod(ACC_PUBLIC, METHOD_NAME_INIT, METHOD_TYPE_INIT,
                      instructions,
                      firstInstruction,
                      lastInstruction);
        }
        else {
            // Find all super-calling constructors.
            Set<Method> constructors =  new HashSet<Method>();
            programClass.methodsAccept(
                new ConstructorMethodFilter(
                new MethodCollector(constructors), null, null));

            if (constructors.size() == 1)
            {
                // There is only one supper-calling constructor.
                // Add the code to this constructor.
                this.instructions = instructions;
                constructors.iterator().next().accept(programClass,
                    new AllAttributeVisitor(
                    this));
            }
            else
            {
                // There are multiple super-calling constructors. Add the
                // instructions to a separate, parameterless initialization
                // method, and invoke this method from all super-calling
                // constructors.
                ProgramMethod initMethod = (ProgramMethod) programClass.findMethod(EXTRA_INIT_METHOD_NAME,
                                                                                   EXTRA_INIT_METHOD_DESCRIPTOR);
                if (initMethod == null)
                {
                    // There is no init$ method yet. Create it now, and add the
                    // given instructions to it.
                    initMethod = addMethod(ACC_PRIVATE,
                                           EXTRA_INIT_METHOD_NAME,
                                           EXTRA_INIT_METHOD_DESCRIPTOR,
                                           instructions,
                                           null,
                                           new SimpleInstruction(InstructionConstants.OP_RETURN));

                    // Insert a call to the new init$ method in all super-calling constructors.
                    InstructionSequenceBuilder builder = new InstructionSequenceBuilder(programClass);
                    builder.aload_0();
                    builder.invokespecial(programClass.getName(),
                                           EXTRA_INIT_METHOD_NAME,
                                           EXTRA_INIT_METHOD_DESCRIPTOR,
                                           programClass,
                                           initMethod);
                    this.instructions = builder.instructions();
                    programClass.methodsAccept(
                        new ConstructorMethodFilter(
                        new AllAttributeVisitor(
                        this), null, null));
                }
                else {
                    // There is already an init$ method. Add the instructions to this method.
                    this.instructions = instructions;
                    initMethod.accept(programClass,
                        new AllAttributeVisitor(
                        this));
                }
            }
        }
    }


    /**
     * Adds a new method to the edited class, with the given instructions array.
     *
     * @param u2accessFlags         acces flags for the new method.
     * @param methodName            name of the new method.
     * @param methodDescriptor      descriptor of the new method.
     * @param instructions          the instructions of the new method.
     * @param firstInstructions     extra instructions to add in front of the
     *                              new method.
     * @param lastInstruction       extra instruction to add at the end of the
     *                              new method.
     */
    private ProgramMethod addMethod(int           u2accessFlags,
                                    String        methodName,
                                    String        methodDescriptor,
                                    Instruction[] instructions,
                                    Instruction[] firstInstructions,
                                    Instruction   lastInstruction)
    {
        ProgramMethod method = new ProgramMethod(u2accessFlags,
                                                 constantPoolEditor.addUtf8Constant(methodName),
                                                 constantPoolEditor.addUtf8Constant(methodDescriptor),
                                                 null);

        CodeAttribute codeAttribute =
            new CodeAttribute(constantPoolEditor.addUtf8Constant(ClassConstants.ATTR_Code));

        CodeAttributeComposer composer = new CodeAttributeComposer();
        composer.reset();
        composer.beginCodeFragment(0);
        composer.appendInstructions(instructions);
        if (firstInstructions != null) {
            for (Instruction instruction : firstInstructions) {
                composer.appendInstruction(instruction);
            }
        }
        if (lastInstruction != null) {
            composer.appendInstruction(lastInstruction);
        }
        composer.endCodeFragment();
        composer.visitCodeAttribute(programClass, method, codeAttribute);

        new AttributesEditor(programClass, method, false).addAttribute(codeAttribute);

        classEditor.addMethod(method);

        return method;
    }


    // Implementations for AttributeVisitor.

    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        CodeAttributeEditor codeAttributeEditor = new CodeAttributeEditor();
        ((ProgramMethod) method).attributesAccept(programClass, new CodeAttributeEditorResetter(codeAttributeEditor));
        codeAttributeEditor.insertBeforeOffset(0, instructions);
        codeAttributeEditor.visitCodeAttribute(clazz, method, codeAttribute);

    }


    private class CodeComposer extends CompactCodeAttributeComposer {

        private final ProgramMethod method;

        public CodeComposer(ProgramMethod method,
                            int           maxCodeFragmentLength)
        {
            super(programClass);
            this.method = method;
            beginCodeFragment(maxCodeFragmentLength);
        }

        public void finishEditing() {
            endCodeFragment();

            CodeAttribute codeAttribute =
                new CodeAttribute(constantPoolEditor.addUtf8Constant(ClassConstants.ATTR_Code));

            visitCodeAttribute(programClass, method, codeAttribute);

            new AttributesEditor(programClass, method, false).addAttribute(codeAttribute);

            classEditor.addMethod(method);
        }
    }
}
