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
import proguard.classfile.constant.*;
import proguard.classfile.instruction.*;
import proguard.classfile.instruction.visitor.InstructionVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;
import proguard.classfile.VisitorAccepter;

import java.util.Arrays;

/**
 * This ClassVisitor removes all unused entries from the bootstrap method attribute.
 *
 * If all bootstrap methods are removed, it also removes the BootstrapMethodsAttribute from
 * the visited class. Additionally, the java/lang/MethodHandles$Lookup class will be
 * removed from the InnerClasses attribute and the InnerClassesAttribute will be removed if
 * it was the only entry.
 *
 * @author Tim Van Den Broecke
 */
public class BootstrapMethodsAttributeShrinker
extends      SimplifiedVisitor
implements   ClassVisitor,

             // Implementation interfaces.
             MemberVisitor,
             AttributeVisitor,
             InstructionVisitor,
             BootstrapMethodInfoVisitor
{
    // A visitor info flag to indicate the bootstrap method is being used.
    private static final Object USED = new Object();

    private       int[]                   bootstrapMethodIndexMap = new int[ClassConstants.TYPICAL_BOOTSTRAP_METHODS_ATTRIBUTE_SIZE];
    private final BootstrapMethodRemapper bootstrapMethodRemapper = new BootstrapMethodRemapper(true);

    private int     referencedBootstrapMethodIndex = -1;
    private boolean modified                       = false;


    // Implementations for ClassVisitor.

    @Override
    public void visitLibraryClass(LibraryClass libaryClass) {}


    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        // Clear the fields from any previous runs.
        modified                = false;
        bootstrapMethodIndexMap = new int[ClassConstants.TYPICAL_BOOTSTRAP_METHODS_ATTRIBUTE_SIZE];

        // Remove any previous visitor info.
        programClass.accept(new ClassCleaner());

        // Mark the bootstrap methods referenced by invokeDynamic instructions.
        programClass.methodsAccept(this);

        // Shrink the bootstrap methods attribute
        programClass.attributesAccept(this);

        if (modified)
        {
            // Clean up dangling and freed up constants
            programClass.accept(new ConstantPoolShrinker());
        }
    }


    // Implementations for MemberVisitor.

    @Override
    public void visitProgramMethod(ProgramClass programClass, ProgramMethod programMethod)
    {
        programMethod.attributesAccept(programClass, this);
    }


    // Implementations for AttributeVisitor.

    @Override
    public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}


    @Override
    public void visitCodeAttribute(Clazz clazz, Method method, CodeAttribute codeAttribute)
    {
        codeAttribute.instructionsAccept(clazz, method, this);
    }


    @Override
    public void visitBootstrapMethodsAttribute(Clazz clazz, BootstrapMethodsAttribute bootstrapMethodsAttribute)
    {
        if (referencedBootstrapMethodIndex > -1)
        {
            // We're marking bootstrap methods
            bootstrapMethodsAttribute.bootstrapMethodEntryAccept(clazz, referencedBootstrapMethodIndex, this);
        }
        else
        {
            // The bootstrap methods have been marked, so now we shrink the array of BootstrapMethodInfo objects.
            int newBootstrapMethodsCount =
                shrinkBootstrapMethodArray(bootstrapMethodsAttribute.bootstrapMethods,
                                           bootstrapMethodsAttribute.u2bootstrapMethodsCount);

            if (newBootstrapMethodsCount < bootstrapMethodsAttribute.u2bootstrapMethodsCount)
            {
                modified = true;

                bootstrapMethodsAttribute.u2bootstrapMethodsCount = newBootstrapMethodsCount;

                if (bootstrapMethodsAttribute.u2bootstrapMethodsCount == 0)
                {
                    // Remove the entire attribute.
                    AttributesEditor attributesEditor = new AttributesEditor((ProgramClass)clazz, false);
                    attributesEditor.deleteAttribute(ClassConstants.ATTR_BootstrapMethods);

                    // Only bootstrap methods require the java/lang/MethodHandles$Lookup
                    // inner class, so we can remove it.
                    clazz.attributesAccept(new MethodHandlesLookupInnerClassRemover(attributesEditor));
                }
                else
                {
                    // Remap all constant pool references to remaining bootstrap methods.
                    bootstrapMethodRemapper.setBootstrapMethodIndexMap(bootstrapMethodIndexMap);
                    clazz.constantPoolEntriesAccept(bootstrapMethodRemapper);
                }
            }
        }
    }


    // Implementations for InstructionVisitor.

    @Override
    public void visitAnyInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, Instruction instruction) {}


    @Override
    public void visitConstantInstruction(Clazz clazz, Method method, CodeAttribute codeAttribute, int offset, ConstantInstruction constantInstruction)
    {
        if (constantInstruction.opcode == InstructionConstants.OP_INVOKEDYNAMIC)
        {
            ProgramClass programClass = (ProgramClass)clazz;

            InvokeDynamicConstant invokeDynamicConstant =
                (InvokeDynamicConstant)programClass.getConstant(constantInstruction.constantIndex);

            referencedBootstrapMethodIndex = invokeDynamicConstant.getBootstrapMethodAttributeIndex();

            programClass.attributesAccept(this);

            referencedBootstrapMethodIndex = -1;
        }
    }


    // Implementations for BootstrapMethodInfoVisitor.

    @Override
    public void visitBootstrapMethodInfo(Clazz clazz, BootstrapMethodInfo bootstrapMethodInfo)
    {
        markAsUsed(bootstrapMethodInfo);
    }


    // Small utility methods.

    /**
     * Marks the given visitor accepter as being used.
     */
    private void markAsUsed(BootstrapMethodInfo bootstrapMethodInfo)
    {
        bootstrapMethodInfo.setVisitorInfo(USED);
    }


    /**
     * Returns whether the given visitor accepter has been marked as being used.
     */
    private boolean isUsed(VisitorAccepter visitorAccepter)
    {
        return visitorAccepter.getVisitorInfo() == USED;
    }


    /**
     * Removes all entries that are not marked as being used from the given
     * array of bootstrap methods. Creates a map from the old indices to the
     * new indices as a side effect.
     * @return the new number of entries.
     */
    private int shrinkBootstrapMethodArray(BootstrapMethodInfo[] bootstrapMethods, int length)
    {
        if (bootstrapMethodIndexMap.length < length)
        {
            bootstrapMethodIndexMap = new int[length];
        }

        int counter = 0;

        // Shift the used bootstrap methods together.
        for (int index = 0; index < length; index++)
        {
            BootstrapMethodInfo bootstrapMethod = bootstrapMethods[index];

            // Is the entry being used?
            if (isUsed(bootstrapMethod))
            {
                // Remember the new index.
                bootstrapMethodIndexMap[index] = counter;

                // Shift the entry.
                bootstrapMethods[counter++] = bootstrapMethod;
            }
            else
            {
                // Remember an invalid index.
                bootstrapMethodIndexMap[index] = -1;
            }
        }

        // Clear the remaining bootstrap methods.
        Arrays.fill(bootstrapMethods, counter, length, null);

        return counter;
    }

    private class MethodHandlesLookupInnerClassRemover
    extends    SimplifiedVisitor
    implements AttributeVisitor,

               // Implementation interfaces.
               InnerClassesInfoVisitor
    {
        private static final String METHOD_HANDLES_CLASS = "java/lang/invoke/MethodHandles";

        private final Object methodHandleLookupMarker = new Object();

        private final AttributesEditor attributesEditor;

        public MethodHandlesLookupInnerClassRemover(AttributesEditor attributesEditor)
        {
            this.attributesEditor = attributesEditor;
        }

        // Implementations for AttributeVisitor

        public void visitAnyAttribute(Clazz clazz, Attribute attribute) {}

        public void visitInnerClassesAttribute(Clazz clazz, InnerClassesAttribute innerClassesAttribute)
        {
            // Mark inner class infos that refer to Lookup.
            innerClassesAttribute.innerClassEntriesAccept(clazz, this);

            // Remove all marked inner classes.
            InnerClassesAttributeEditor editor =
                new InnerClassesAttributeEditor(innerClassesAttribute);
            for (int index = innerClassesAttribute.u2classesCount - 1; index >= 0; index--)
            {
                InnerClassesInfo innerClassesInfo = innerClassesAttribute.classes[index];
                if (shouldBeRemoved(innerClassesInfo))
                {
                    editor.removeInnerClassesInfo(innerClassesInfo);
                }
            }

            // Remove the attribute if it is empty.
            if (innerClassesAttribute.u2classesCount == 0)
            {
                attributesEditor.deleteAttribute(ClassConstants.ATTR_InnerClasses);
            }
        }


        // Implementations for InnerClassesInfoVisitor.

        public void visitInnerClassesInfo(Clazz clazz, InnerClassesInfo innerClassesInfo)
        {
            ProgramClass programClass = (ProgramClass) clazz;

            ClassConstant innerClass =
                (ClassConstant) programClass.getConstant(innerClassesInfo.u2innerClassIndex);
            ClassConstant outerClass =
                (ClassConstant) programClass.getConstant(innerClassesInfo.u2outerClassIndex);

            if (isMethodHandleClass(innerClass, clazz) ||
                isMethodHandleClass(outerClass, clazz))
            {
                markForRemoval(innerClassesInfo);
            }
        }


        // Small utility methods.

        private void markForRemoval(InnerClassesInfo innerClassesInfo)
        {
            innerClassesInfo.setVisitorInfo(methodHandleLookupMarker);
        }

        private boolean shouldBeRemoved(InnerClassesInfo innerClassesInfo)
        {
            return innerClassesInfo.getVisitorInfo() == methodHandleLookupMarker;
        }

        public boolean isMethodHandleClass(ClassConstant classConstant, Clazz clazz)
        {
            return classConstant != null &&
                   classConstant.getName(clazz).startsWith(METHOD_HANDLES_CLASS);
        }
    }
}
