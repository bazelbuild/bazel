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
import proguard.classfile.attribute.annotation.*;
import proguard.classfile.attribute.annotation.visitor.*;
import proguard.classfile.constant.*;
import proguard.classfile.constant.visitor.ConstantVisitor;
import proguard.classfile.util.SimplifiedVisitor;
import proguard.classfile.visitor.*;

import java.util.*;

/**
 * This class and member visitor collects the classes and fields that can be
 * involved in Json (de)serialization and register their Java to Json field
 * name mapping in an OptimizedJsonInfo object
 *
 * @author Lars Vandenbergh
 * @author Rob Coekaerts
 */
public class OptimizedJsonFieldCollector
extends      SimplifiedVisitor
implements   ClassVisitor,
             MemberVisitor
{
    private final OptimizedJsonInfo               optimizedJsonInfo;
    private final Mode                            mode;
    private       OptimizedJsonInfo.ClassJsonInfo classJsonInfo;


    /**
     * Creates a new OptimizedJsonFieldCollector.
     *
     * @param optimizedJsonInfo contains information on which classes and fields
     *                          need to optimized and how.
     * @param mode              whether serialization or deserialization is
     *                          being done.
     */
    public OptimizedJsonFieldCollector(OptimizedJsonInfo optimizedJsonInfo,
                                       Mode              mode)
    {
        this.optimizedJsonInfo = optimizedJsonInfo;
        this.mode              = mode;
    }


    // Implementations for ClassVisitor.

    @Override
    public void visitProgramClass(ProgramClass programClass)
    {
        classJsonInfo = new OptimizedJsonInfo.ClassJsonInfo();
        optimizedJsonInfo.classJsonInfos.put(programClass.getName(), classJsonInfo);
        optimizedJsonInfo.classIndices.put(programClass.getName(), null);
    }


    @Override
    public void visitLibraryClass(LibraryClass libraryClass) {}


    // Implementations for MemberVisitor.

    @Override
    public void visitAnyMember(Clazz clazz, Member member) {}


    @Override
    public void visitProgramField(ProgramClass programClass, ProgramField programField)
    {
        OptimizedJsonInfo.ClassJsonInfo classJsonInfo =
            optimizedJsonInfo.classJsonInfos.get(programClass.getName());

        programField.attributesAccept(programClass,
            new AllAnnotationVisitor(
            new MultiAnnotationVisitor(
                new AnnotationTypeFilter(GsonClassConstants.ANNOTATION_TYPE_SERIALIZED_NAME,
                new SerializedNamesCollector(classJsonInfo)),
                new AnnotationTypeFilter(GsonClassConstants.ANNOTATION_TYPE_EXPOSE,
                new ExposedFieldsCollector(classJsonInfo, mode)))));


        String fieldName = programField.getName(programClass);
        if (classJsonInfo.javaToJsonFieldNames.get(fieldName) == null)
        {
            classJsonInfo.javaToJsonFieldNames.put(fieldName, new String[] { fieldName });
            optimizedJsonInfo.jsonFieldIndices.put(fieldName, null);
        }
        else
        {
            for (String jsonFieldName: classJsonInfo.javaToJsonFieldNames.get(fieldName))
            {
                optimizedJsonInfo.jsonFieldIndices.put(jsonFieldName, null);
            }
        }
    }


    private static class ExposedFieldsCollector
    extends              SimplifiedVisitor
    implements           AnnotationVisitor,
                         ElementValueVisitor,
                         ConstantVisitor
    {
        private final OptimizedJsonInfo.ClassJsonInfo classJsonInfo;
        private final Mode                            mode;

        public boolean exposeCurrentField;


        public ExposedFieldsCollector(OptimizedJsonInfo.ClassJsonInfo classJsonInfo, Mode mode)
        {
            this.classJsonInfo = classJsonInfo;
            this.mode          = mode;
        }


        // Implementations for AnnotationVisitor

        @Override
        public void visitAnnotation(Clazz clazz, Annotation annotation) {}


        @Override
        public void visitAnnotation(Clazz clazz, Field field, Annotation annotation)
        {
            exposeCurrentField = true;
            annotation.elementValuesAccept(clazz, this);
            if (exposeCurrentField)
            {
                classJsonInfo.exposedJavaFieldNames.add(field.getName(clazz));
            }
        }


        // Implementations for ElementValueVisitor

        @Override
        public void visitAnyElementValue(Clazz clazz, Annotation annotation, ElementValue elementValue) {}


        @Override
        public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
        {
            if(constantElementValue.getMethodName(clazz).equals(mode.toString()))
            {
                clazz.constantPoolEntryAccept(constantElementValue.u2constantValueIndex, this);
            }
        }


        @Override
        public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue) {}


        // Implementations for ConstantVisitor

        @Override
        public void visitAnyConstant(Clazz clazz, Constant constant) {}


        @Override
        public void visitIntegerConstant(Clazz clazz, IntegerConstant integerConstant)
        {
            if (integerConstant.u4value == 0)
            {
                exposeCurrentField = false;
            }
        }
    }


    private static class SerializedNamesCollector
    extends              SimplifiedVisitor
    implements           AnnotationVisitor,
                         ElementValueVisitor
    {
        private final OptimizedJsonInfo.ClassJsonInfo classJsonInfo;
        private       List<String>                    currentJsonFieldNames;


        public SerializedNamesCollector(OptimizedJsonInfo.ClassJsonInfo classJsonInfo)
        {
            this.classJsonInfo = classJsonInfo;
        }


        // Implementations for AnnotationVisitor

        @Override
        public void visitAnnotation(Clazz clazz, Annotation annotation) {}


        @Override
        public void visitAnnotation(Clazz clazz, Field field, Annotation annotation)
        {
            currentJsonFieldNames = new ArrayList<String>();

            annotation.elementValuesAccept(clazz, this);

            String[] jsonNamesArray = currentJsonFieldNames.toArray(new String[0]);
            classJsonInfo.javaToJsonFieldNames.put(field.getName(clazz), jsonNamesArray);
        }


        // Implementations for ElementValueVisitor

        @Override
        public void visitAnyElementValue(Clazz clazz, Annotation annotation, ElementValue elementValue) {}


        @Override
        public void visitConstantElementValue(Clazz clazz, Annotation annotation, ConstantElementValue constantElementValue)
        {
            currentJsonFieldNames.add(clazz.getString(constantElementValue.u2constantValueIndex));
        }


        @Override
        public void visitArrayElementValue(Clazz clazz, Annotation annotation, ArrayElementValue arrayElementValue)
        {
            arrayElementValue.elementValuesAccept(clazz, annotation, this);
        }
    }


    public enum Mode
    {
        serialize,
        deserialize
    }
}
