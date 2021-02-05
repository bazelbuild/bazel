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
package proguard.optimize.gson;

import proguard.classfile.*;
import proguard.classfile.attribute.SignatureAttribute;
import proguard.classfile.editor.*;
import proguard.optimize.info.ProgramClassOptimizationInfoSetter;

import static proguard.classfile.ClassConstants.ACC_PUBLIC;
import static proguard.classfile.ClassConstants.CLASS_VERSION_1_5;

/**
 * This builder builds a GSON TypeToken class based on a given field signature.
 *
 * @author Lars Vandenbergh
 */
class TypeTokenClassBuilder
{
    private final ProgramClass programClass;
    private final ProgramField programField;
    private final String       fieldSignature;


    /**
     * Creates a new TypeTokenClassBuilder.
     *
     * @param programClass   the class containing the field for which to build
     *                       a type token.
     * @param programField   the field for which to build a type token.
     * @param fieldSignature the signature of the field.
     */
    public TypeTokenClassBuilder(ProgramClass programClass,
                                 ProgramField programField,
                                 String       fieldSignature) {

        this.programClass   = programClass;
        this.programField   = programField;
        this.fieldSignature = fieldSignature;
    }


    /**
     * Builds and returns a new TypeToken subclass that contains all necessary
     * type information in its signature.
     *
     * @param programClassPool the program class pool used to look up class
     *                         references.
     * @return the TypeToken class with the correct signature.
     */
    public ProgramClass build(ClassPool programClassPool)
    {
        String typeTokenClassName = programClass.getName() +
                                    programField.getName(programClass) +
                                    "TypeToken";

        // Create sub-class of TypeToken with default constructor.
        SimplifiedClassEditor classEditor =
            new SimplifiedClassEditor(ClassConstants.ACC_PUBLIC,
                                      typeTokenClassName,
                                      GsonClassConstants.NAME_TYPE_TOKEN);
        classEditor.addMethod(ACC_PUBLIC,
                              ClassConstants.METHOD_NAME_INIT,
                              ClassConstants.METHOD_TYPE_INIT,
                              10)
                   .aload_0()
                   .invokespecial(GsonClassConstants.NAME_TYPE_TOKEN,
                                  ClassConstants.METHOD_NAME_INIT,
                                  ClassConstants.METHOD_TYPE_INIT)
                   .return_();

        classEditor.finishEditing();

        ProgramClass subClass = classEditor.getProgramClass();
        subClass.accept(new ProgramClassOptimizationInfoSetter(true));
        programClassPool.classAccept(GsonClassConstants.NAME_TYPE_TOKEN,
                                     new SubclassAdder(subClass));

        // Class version 1.5 is required to make the Java runtime even consider
        // the Signature attribute.
        subClass.u4version = CLASS_VERSION_1_5;

        // Add signature attribute with full generic type.
        ConstantPoolEditor constantPoolEditor = new ConstantPoolEditor(subClass);
        int                attributeNameIndex = constantPoolEditor.addUtf8Constant("Signature");
        String             classSignature     = "Lcom/google/gson/reflect/TypeToken<" + this.fieldSignature + ">;";
        int                signatureIndex     = constantPoolEditor.addUtf8Constant(classSignature);
        AttributesEditor   attributesEditor   = new AttributesEditor(subClass, false);
        attributesEditor.addAttribute(new SignatureAttribute(attributeNameIndex, signatureIndex));

        return subClass;
    }
}
